// Modifying this for aerodynamics 
// bunny3d_demo - Interactive orbit/pan camera test
// Usage: bunny3d_demo [--terminal] [--braille] [model.xml]
// Controls:
//   Left-Drag          = Orbit camera
//   Middle/Right-Drag  = Pan camera
//   Arrow keys         = Pan camera
//   Scroll / PgUp/PgDn = Zoom
//   Space              = Pause/Resume simulation
//   B                  = Toggle braille mode
//   R                  = Restart simulation
//   W                  = Toggle wireframe
//   < / >              = Slow down / speed up physics (1/16x to 16x)
//   , / .              = Step backward / forward (pauses)
//   ESC / Ctrl+C       = Quit

#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700

#include <GLES2/gl2.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <math.h>
#include <mujoco/mujoco.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#ifndef __USECONDS_T_DEFINED
#define __USECONDS_T_DEFINED
typedef unsigned int useconds_t;
#endif
#ifndef __USLEEP_DECLARED
#define __USLEEP_DECLARED 1
extern int usleep(useconds_t);
#endif

#include <pollyn/input.h>
#include <pollyn/line_renderer.h>
#include <pollyn/math_util.h>
#include <pollyn/renderer.h>
#include <pollyn/terminal_view.h>
#include <pollyn/text_overlay.h>
#include <pollyn/virtual_framebuffer.h>
#include <trellis/plugins/mujoco/mujoco_bridge.h>

// -----------------------------------------------------------------------------
// Keyboard → actuator mapping
// -----------------------------------------------------------------------------
// The mapping below associates a GLFW key code with an actuator index and a
// delta value to apply to the actuator's control signal.  The array is
// declared `static const` so it can be modified by the user when editing the
// source, but it is not altered at runtime.  Add or remove entries as needed.
//
// Example: map the 'w' key to increase the first actuator by 0.1, and the
// 's' key to decrease it by 0.1.
// -----------------------------------------------------------------------------
typedef struct {
	int key;	    // GLFW key code
	int actuator_index; // index into d->ctrl array
	float delta;	    // amount to add/subtract per key press
} KeyActuatorMap;

// Global buffer holding the current control value for each actuator.
// Size 64 is arbitrary but should cover most models.  Values are zeroed on
// startup and updated by keyboard input.
static float actuator_inputs[64] = {0.0f};

static const KeyActuatorMap key_actuator_map[] = {
		/* First actuator (rear left wheel) */
		{GLFW_KEY_Y, 0, 0.5f},
		{GLFW_KEY_H, 0, -0.5f},
		/* Second actuator (rear right wheel) */
		{GLFW_KEY_I, 1, 0.5f},
		{GLFW_KEY_K, 1, -0.5f},
};

//inserted
// --- AeroSandbox VLM Adaptation: Start ---
typedef struct {
    int geom_id;
    double cl_alpha;    // Lift slope (from ASB VLM)
    double cl0;         // Zero-alpha lift
    double cd0;         // Parasitic drag (profile)
    double k_induced;   // Induced drag factor (1 / (pi * AR * e))
    double cm_alpha;    // Pitching moment slope
    double cm0;         // Zero-alpha moment
    double area;        // MAC * Span
    double chord;       // Mean Aerodynamic Chord (for moments)
} VLMAeroProperties;

static VLMAeroProperties wings[16];
static int num_wings = 0;

//inserted
void aerodynamics_callback(const mjModel* m, mjData* d) {
    for (int i = 0; i < num_wings; i++) {
        VLMAeroProperties* aero = &wings[i];
        int gid = aero->geom_id;

        mjtNum vel_world[6], local_vel[3];
        mj_objectVelocity(m, d, mjOBJ_GEOM, gid, vel_world, 0);
        mju_mulMatTVec3(local_vel, d->geom_xmat + 9 * gid, vel_world + 3);

        double V = mju_norm3(local_vel) + 1e-6;
        // AoA calculation: note the sign depends on your XML orientation
        double alpha = atan2(-local_vel[2], local_vel[0]); 
        
        // Dynamic Pressure * Area
        double q_s = 0.5 * 1.225 * V * V * aero->area;

        // VLM Aero Equations
        double cl = aero->cl0 + (aero->cl_alpha * alpha);
        double cd = aero->cd0 + (aero->k_induced * cl * cl);
        double cm = aero->cm0 + (aero->cm_alpha * alpha);

        // Forces in Local Frame (Lift is Z, Drag is -X)
        mjtNum f_local[3] = {-cd * q_s, 0, cl * q_s};
        // Pitching Moment (about Y axis)
        mjtNum m_local[3] = {0, cm * q_s * aero->chord, 0}; 
        
        mjtNum f_world[3], m_world[3];
        mju_mulMatVec3(f_world, d->geom_xmat + 9 * gid, f_local);
        mju_mulMatVec3(m_world, d->geom_xmat + 9 * gid, m_local);

        // Apply to the specific body the wing is attached to
        mj_applyFT(m, d, f_world, m_world, d->geom_xpos + 3 * gid, m->geom_bodyid[gid], d->qfrc_passive);
    }
}
// --- AeroSandbox VLM Adaptation: End ---
//inserted

typedef struct {
	float rot[9]; // 3x3 rotation matrix (column-major): columns are right, up, back
	float distance;
	float target[3];
} Camera;

// Multiply 3x3 matrix by vector: out = M * v
static void mat3_mul_vec3(float out[3], const float m[9], const float v[3]) {
	out[0] = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
	out[1] = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
	out[2] = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
}

// Multiply two 3x3 matrices: out = a * b (column-major)
static void mat3_mul(float out[9], const float a[9], const float b[9]) {
	for(int col = 0; col < 3; col++) {
		for(int row = 0; row < 3; row++) {
			out[col * 3 + row] = a[0 * 3 + row] * b[col * 3 + 0] + a[1 * 3 + row] * b[col * 3 + 1] + a[2 * 3 + row] * b[col * 3 + 2];
		}
	}
}

// Create rotation matrix from axis-angle
static void mat3_from_axis_angle(float out[9], const float axis[3], float angle) {
	float c = cosf(angle);
	float s = sinf(angle);
	float t = 1.0f - c;
	float x = axis[0], y = axis[1], z = axis[2];

	out[0] = t * x * x + c;
	out[3] = t * x * y - s * z;
	out[6] = t * x * z + s * y;
	out[1] = t * x * y + s * z;
	out[4] = t * y * y + c;
	out[7] = t * y * z - s * x;
	out[2] = t * x * z - s * y;
	out[5] = t * y * z + s * x;
	out[8] = t * z * z + c;
}

// Set matrix to identity
static void mat3_identity(float m[9]) {
	m[0] = 1;
	m[3] = 0;
	m[6] = 0;
	m[1] = 0;
	m[4] = 1;
	m[7] = 0;
	m[2] = 0;
	m[5] = 0;
	m[8] = 1;
}

static void camera_init(Camera *cam) {
	// Initialize with a tilt looking at target from above
	// Start looking from positive X+Y direction, higher angle
	float yaw = 0.5f;
	float pitch = 0.5f;

	// Build rotation from yaw/pitch for initial orientation
	float cy = cosf(yaw), sy = sinf(yaw);
	float cp = cosf(pitch), sp = sinf(pitch);

	// Forward direction (camera looks along -Z in camera space, so this is the "back" direction)
	float fwd[3] = {cp * cy, cp * sy, sp};
	vec3_normalize(fwd, fwd);

	// Right = fwd x world_up
	float world_up[3] = {0, 0, 1};
	float right[3], up[3];
	vec3_cross(right, fwd, world_up);
	vec3_normalize(right, right);
	vec3_cross(up, right, fwd);

	// Store as column-major: col0=right, col1=up, col2=back(fwd)
	cam->rot[0] = right[0];
	cam->rot[1] = right[1];
	cam->rot[2] = right[2];
	cam->rot[3] = up[0];
	cam->rot[4] = up[1];
	cam->rot[5] = up[2];
	cam->rot[6] = fwd[0];
	cam->rot[7] = fwd[1];
	cam->rot[8] = fwd[2];

	cam->distance = 4.0f;
	cam->target[0] = 0.0f;
	cam->target[1] = 0.0f;
	cam->target[2] = 1.0f;
}

// Get camera axes from rotation matrix
static void camera_axes(const Camera *cam, float right[3], float up[3], float fwd[3]) {
	// Column 0 = right, Column 1 = up, Column 2 = forward (back direction)
	right[0] = cam->rot[0];
	right[1] = cam->rot[1];
	right[2] = cam->rot[2];
	up[0] = cam->rot[3];
	up[1] = cam->rot[4];
	up[2] = cam->rot[5];
	fwd[0] = cam->rot[6];
	fwd[1] = cam->rot[7];
	fwd[2] = cam->rot[8];
}

// Rotate camera around an axis
static void camera_rotate(Camera *cam, const float axis[3], float angle) {
	float rot[9], result[9];
	mat3_from_axis_angle(rot, axis, angle);
	mat3_mul(result, rot, cam->rot);
	memcpy(cam->rot, result, sizeof(cam->rot));
}

static void camera_matrices(const Camera *cam, float aspect, float view[16], float proj[16],
			    float vp[16], float eye[3]) {
	float right[3], up[3], fwd[3];
	camera_axes(cam, right, up, fwd);

	// Eye position: target + fwd * distance (fwd points away from target)
	vec3_scale(eye, fwd, cam->distance);
	vec3_add(eye, cam->target, eye);

	// Look at target from eye
	mat4_lookat(view, eye, cam->target, up);
	mat4_perspective(proj, 45.0f, aspect, 0.01f, 100.0f);
	mat4_mul(vp, proj, view);
}

static InputState g_input;

static void key_cb(GLFWwindow *w, int key, int sc, int act, int mods) {
	input_on_key(&g_input, key, act, mods);
}
static void char_cb(GLFWwindow *w, unsigned int codepoint) {
	input_on_char(&g_input, codepoint);
}
static void mouse_cb(GLFWwindow *w, int btn, int act, int mods) {
	input_on_mouse_button(&g_input, btn, act);
}
static void cursor_cb(GLFWwindow *w, double x, double y) {
	input_on_cursor_pos(&g_input, x, y);
}
static void scroll_cb(GLFWwindow *w, double dx, double dy) {
	input_on_scroll(&g_input, dx, dy);
}

int main(int argc, char **argv) {
	const char *model_path = "testdata/bunny.xml";
	int terminal_mode = 0;
	int braille_mode = 0;

	for(int i = 1; i < argc; i++) {
		if(strcmp(argv[i], "--terminal") == 0) {
			terminal_mode = 1;
		} else if(strcmp(argv[i], "--braille") == 0) {
			braille_mode = 1;
			terminal_mode = 1; // braille implies terminal mode
		} else if(argv[i][0] != '-') {
			model_path = argv[i];
		}
	}

	char err[512] = {0};
	mjModel *m = mj_loadXML(model_path, NULL, err, sizeof(err));
	if(!m) {
		fprintf(stderr, "Load error: %s\n", err);
		return 1;
	}
	mjData *d = mj_makeData(m);
	// --- INSERT THIS ---
    mjcb_passive = aerodynamics_callback; 
    // -------------------
	mj_forward(m, d); // Compute initial transforms

// --- INSERTED BLOCK #3: Aero Initialization ---
    // Change "wing_geom" to the name of the geom in your XML file
// --- VLM Initialization Logic ---
    int wid = mj_name2id(m, mjOBJ_GEOM, "wing_geom");
    if (wid != -1) {
        wings[num_wings].geom_id = wid;
        
        // These values come from your ASB VLM solve:
        wings[num_wings].cl_alpha  = 4.5;    // VLM Lift slope
        wings[num_wings].cl0       = 0.2;    // Lift at 0 AoA
        wings[num_wings].cd0       = 0.01;   // Profile drag
        wings[num_wings].k_induced = 0.04;   // VLM-calculated induced drag k
        wings[num_wings].cm_alpha  = -0.5;   // Longitudinal stability
        wings[num_wings].cm0       = -0.02;  // Moment at 0 AoA
        
        wings[num_wings].area      = 0.5;    // S ref
        wings[num_wings].chord     = 0.2;    // c bar
        num_wings++;
    }
    // ----------------------------------------------
    //inserted

	if(!glfwInit()) {
		return 1;
	}
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	if(terminal_mode) {
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	}

	int win_w = terminal_mode ? 640 : 800;
	int win_h = terminal_mode ? 360 : 600;
	GLFWwindow *win = glfwCreateWindow(win_w, win_h, "Bunny3D Demo", NULL, NULL);
	if(!win) {
		glfwTerminate();
		return 1;
	}
	glfwMakeContextCurrent(win);
	glfwSwapInterval(1);

	glfwSetKeyCallback(win, key_cb);
	glfwSetCharCallback(win, char_cb);
	glfwSetMouseButtonCallback(win, mouse_cb);
	glfwSetCursorPosCallback(win, cursor_cb);
	glfwSetScrollCallback(win, scroll_cb);

	input_init(&g_input, terminal_mode);

	TerminalView term;
	memset(&term, 0, sizeof(term));
	int fbw, fbh;
	glfwGetFramebufferSize(win, &fbw, &fbh);
	int last_fbw = fbw;
	int last_fbh = fbh;

	// Create virtual framebuffer for offscreen rendering
	// Use VFB for both terminal mode (for readback) and windowed mode (for consistency)
	VirtualFramebuffer vfb;
	int use_vfb = 1; // Always use VFB for reliable rendering
	if(use_vfb) {
		if(!vfb_create(&vfb, fbw, fbh, 1)) { // with depth buffer
			fprintf(stderr, "Failed to create virtual framebuffer\n");
			return 1;
		}
	}

	if(terminal_mode) {
		terminal_view_init(&term, 0, 0, 0, 0, 0, braille_mode, 0);
	}

	MuJoCoScene scene;
	if(!mujoco_scene_init(&scene, m)) {
		fprintf(stderr, "Scene init failed\n");
		return 1;
	}
	mujoco_scene_update(&scene, m, d);

	Renderer *ren = renderer_create(fbw, fbh);
	if(!ren) {
		fprintf(stderr, "Failed to create renderer\n");
		return 1;
	}

	LineRenderer *line_ren = line_renderer_create();
	if(!line_ren) {
		fprintf(stderr, "Failed to create line renderer\n");
		renderer_destroy(ren);
		return 1;
	}

// State history buffer for stepping backward
#define STATE_HISTORY_SIZE 512
	mjData **state_history = (mjData **) calloc(STATE_HISTORY_SIZE, sizeof(mjData *));
	for(int i = 0; i < STATE_HISTORY_SIZE; i++) {
		state_history[i] = mj_makeData(m);
	}
	int history_head = 0;  // Next position to write in ring buffer
	int history_count = 0; // Number of valid states in buffer

	// Save initial state
	mj_copyData(state_history[0], m, d);
	history_head = 1;
	history_count = 1;

	RendererLight lights[RENDERER_MAX_LIGHTS];
	int nlights = mujoco_scene_lights(m, d, lights, RENDERER_MAX_LIGHTS);
	renderer_set_lights(ren, lights, nlights);

	Camera cam;
	camera_init(&cam);

	TextOverlay text;
	text_overlay_init(&text);

	double last_time = glfwGetTime();
	double frame_time_ms = 0.0;
	double physics_time_ms = 0.0;  // Time spent in physics this frame
	double physics_rt_ratio = 0.0; // Real-time ratio (sim time / wall time)
	int paused = 0;
	int wireframe = 0;
	float speed_multiplier = 1.0f;	   // Physics speed: 0.25x, 0.5x, 1x, 2x, 4x
	double sim_time_accumulator = 0.0; // Accumulated time for physics stepping

	if(!terminal_mode) {
		printf("Controls:\n");
		printf("  Left-Drag      = Orbit\n");
		printf("  Middle/Right   = Pan\n");
		printf("  Arrow keys     = Pan\n");
		printf("  Scroll         = Zoom\n");
		printf("  Space          = Pause/Resume\n");
		printf("  B              = Toggle braille\n");
		printf("  R              = Restart\n");
		printf("  W              = Toggle wireframe\n");
		printf("  < / >          = Slow/Fast physics\n");
		printf("  , / .          = Step back/fwd\n");
		printf("  ESC            = Quit\n");
	}

	while(!glfwWindowShouldClose(win)) {
		double frame_start = glfwGetTime();
		double dt_since_last = frame_start - last_time;

		glfwPollEvents();
		input_poll(&g_input);

		if(input_key_pressed(&g_input, IKEY_ESCAPE) || input_key_pressed(&g_input, IKEY_CTRL_C)) {
			glfwSetWindowShouldClose(win, 1);
		}

		// Toggle pause with spacebar
		if(input_key_pressed(&g_input, IKEY_SPACE)) {
			paused = !paused;
			sim_time_accumulator = 0.0; // Reset to avoid time jump on resume
		}

		// Toggle braille mode with 'b'
		if(input_key_pressed(&g_input, GLFW_KEY_B)) {
			term.braille_mode = !term.braille_mode;
		}

		// Restart simulation with 'r'
		if(input_key_pressed(&g_input, GLFW_KEY_R)) {
			mj_resetData(m, d);
			mj_forward(m, d);
			mujoco_scene_update(&scene, m, d);
			// Reset history buffer
			mj_copyData(state_history[0], m, d);
			history_head = 1;
			history_count = 1;
			sim_time_accumulator = 0.0; // Reset accumulator
			// Zero the global actuator input buffer to start from rest
			for(int a = 0; a < 64; ++a) {
				actuator_inputs[a] = 0.0f;
			}
		}

		// Toggle wireframe with 'w'
		if(input_key_pressed(&g_input, GLFW_KEY_W)) {
			wireframe = !wireframe;
		}

		// Speed up physics with '>' (faster = more steps)
		if(input_char_pressed(&g_input, '>')) {
			speed_multiplier *= 2.0f;
			if(speed_multiplier > 16.0f) {
				speed_multiplier = 16.0f;
			}
			sim_time_accumulator = 0.0;
		}
		// Slow down physics with '<' (slower = fewer steps)
		if(input_char_pressed(&g_input, '<')) {
			speed_multiplier *= 0.5f;
			if(speed_multiplier < 0.0625f) {
				speed_multiplier = 0.0625f; // 1/16x
			}
			sim_time_accumulator = 0.0;
		}
		// Step forward with '.'
		if(input_char_pressed(&g_input, '.')) {
			paused = 1;
			sim_time_accumulator = 0.0;
			mj_step(m, d);
			// Save state to history ring buffer
			mj_copyData(state_history[history_head], m, d);
			history_head = (history_head + 1) % STATE_HISTORY_SIZE;
			if(history_count < STATE_HISTORY_SIZE) {
				history_count++;
			}
			mujoco_scene_update(&scene, m, d);
		}
		// Step backward with ',' (restore from history ring buffer)
		if(input_char_pressed(&g_input, ',')) {
			paused = 1;
			sim_time_accumulator = 0.0;
			if(history_count > 1) {
				// Move back in ring buffer
				history_head = (history_head - 1 + STATE_HISTORY_SIZE) % STATE_HISTORY_SIZE;
				history_count--;
				// Restore state at previous position
				int restore_idx = (history_head - 1 + STATE_HISTORY_SIZE) % STATE_HISTORY_SIZE;
				mj_copyData(d, m, state_history[restore_idx]);
				mujoco_scene_update(&scene, m, d);
			}
		}

		// ---------------------------------------------------------------------
		// Actuator control via keyboard
		// ---------------------------------------------------------------------
		// Update the global actuator input buffer based on key presses and
		// apply a natural decay to all actuators when no key is held.
		const float DECAY_RATE = 0.0f; // units per second towards zero
		const int map_count = (int) (sizeof(key_actuator_map) / sizeof(key_actuator_map[0]));
		for(int i = 0; i < map_count; ++i) {
			if(input_key_pressed(&g_input, key_actuator_map[i].key)) {
				int idx = key_actuator_map[i].actuator_index;
				if(idx >= 0 && idx < m->nu) {
					float cur = actuator_inputs[idx];
					float new_val = cur + key_actuator_map[i].delta * dt_since_last;
					// Clamp to actuator limits
					float lo = m->actuator_ctrlrange[2 * idx];
					float hi = m->actuator_ctrlrange[2 * idx + 1];
					actuator_inputs[idx] = mju_clip(new_val, lo, hi);
				}
			}
		}

		// Decay actuators that are not actively driven
		for(int a = 0; a < m->nu && a < 64; ++a) {
			int any_key = 0;
			for(int i = 0; i < map_count; ++i) {
				if(key_actuator_map[i].actuator_index == a && input_key_pressed(&g_input, key_actuator_map[i].key)) {
					any_key = 1;
					break;
				}
			}
			if(!any_key) {
				float cur = actuator_inputs[a];
				float decay = DECAY_RATE * dt_since_last;
				if(cur > 0.0f) {
					cur -= decay;
					if(cur < 0.0f) {
						cur = 0.0f;
					}
				} else if(cur < 0.0f) {
					cur += decay;
					if(cur > 0.0f) {
						cur = 0.0f;
					}
				}
				actuator_inputs[a] = cur;
			}
		}

		// Copy the global input buffer into the MuJoCo data structure before
		// stepping the simulation.  This ensures that the physics engine sees
		// the updated control signals.
		for(int i = 0; i < m->nu && i < 64; ++i) {
			d->ctrl[i] = actuator_inputs[i];
		}

		// Arrow keys for pan
		{
			float pan_speed = cam.distance * 0.02f;
			float right[3], up[3], fwd[3];
			camera_axes(&cam, right, up, fwd);
			if(input_key_pressed(&g_input, IKEY_LEFT)) {
				float delta[3];
				vec3_scale(delta, right, pan_speed);
				vec3_add(cam.target, cam.target, delta);
			}
			if(input_key_pressed(&g_input, IKEY_RIGHT)) {
				float delta[3];
				vec3_scale(delta, right, -pan_speed);
				vec3_add(cam.target, cam.target, delta);
			}
			if(input_key_pressed(&g_input, IKEY_UP)) {
				float delta[3];
				vec3_scale(delta, up, -pan_speed);
				vec3_add(cam.target, cam.target, delta);
			}
			if(input_key_pressed(&g_input, IKEY_DOWN)) {
				float delta[3];
				vec3_scale(delta, up, pan_speed);
				vec3_add(cam.target, cam.target, delta);
			}
			if(input_key_pressed(&g_input, IKEY_PAGE_UP)) {
				cam.distance *= 0.9f;
				if(cam.distance < 0.5f) {
					cam.distance = 0.5f;
				}
			}
			if(input_key_pressed(&g_input, IKEY_PAGE_DOWN)) {
				cam.distance *= 1.1f;
				if(cam.distance > 20.0f) {
					cam.distance = 20.0f;
				}
			}
		}

		// Mouse drag
		if(g_input.dragging) {
			float dx = g_input.mouse_dx;
			float dy = g_input.mouse_dy;

			if(g_input.button_middle || g_input.button_right) {
				// Pan with middle or right mouse button
				float right[3], up[3], fwd[3];
				camera_axes(&cam, right, up, fwd);
				float scale = cam.distance * (terminal_mode ? 0.03125f : 0.004f);
				float pan[3], r_delta[3], u_delta[3];
				vec3_scale(r_delta, right, dx * scale);
				vec3_scale(u_delta, up, dy * scale);
				vec3_add(pan, r_delta, u_delta);
				vec3_add(cam.target, cam.target, pan);
			} else if(g_input.button_left) {
				// Orbit with left mouse button
				float sens = terminal_mode ? 0.1f : 0.01f;
				float right[3], up[3], fwd[3];
				camera_axes(&cam, right, up, fwd);

				// Rotate around global Z axis for horizontal mouse movement
				float world_up[3] = {0, 0, 1};
				camera_rotate(&cam, world_up, -dx * sens);

				// Rotate around camera's right axis for vertical mouse movement
				camera_axes(&cam, right, up, fwd); // Get updated axes
				camera_rotate(&cam, right, dy * sens);
			}
		}

		// Scroll zoom
		float sx, sy;
		input_get_scroll(&g_input, &sx, &sy);
		if(sy != 0.0f) {
			cam.distance *= expf(-sy * 0.1f);
			if(cam.distance < 0.5f) {
				cam.distance = 0.5f;
			}
			if(cam.distance > 20.0f) {
				cam.distance = 20.0f;
			}
		}

		input_clear_frame(&g_input);

		// Step physics simulation (if not paused)
		if(!paused) {
			// Accumulate time scaled by speed multiplier
			sim_time_accumulator += dt_since_last * speed_multiplier;

			// Step physics while we have enough accumulated time
			double timestep = m->opt.timestep;
			int max_steps = 32; // Cap to prevent stalls (supports up to 16x at 60fps)
			int steps = 0;
			double physics_start = glfwGetTime();
			while(sim_time_accumulator >= timestep && steps < max_steps) {
				mj_step(m, d);
				sim_time_accumulator -= timestep;
				// Save state to history ring buffer
				mj_copyData(state_history[history_head], m, d);
				history_head = (history_head + 1) % STATE_HISTORY_SIZE;
				if(history_count < STATE_HISTORY_SIZE) {
					history_count++;
				}
				steps++;
			}
			double physics_end = glfwGetTime();
			if(steps > 0) {
				physics_time_ms = (physics_end - physics_start) * 1000.0;
				double sim_time_advanced = steps * timestep;
				double wall_time = physics_end - physics_start;
				physics_rt_ratio = (wall_time > 1e-9) ? (sim_time_advanced / wall_time) : 0.0;
				mujoco_scene_update(&scene, m, d);
			}
		}

		if(!terminal_mode) {
			int new_w = 0;
			int new_h = 0;
			glfwGetFramebufferSize(win, &new_w, &new_h);
			if(new_w > 0 && new_h > 0 && (new_w != last_fbw || new_h != last_fbh)) {
				fbw = new_w;
				fbh = new_h;
				last_fbw = new_w;
				last_fbh = new_h;
				renderer_resize(ren, fbw, fbh);
				if(use_vfb) {
					vfb_resize(&vfb, fbw, fbh);
				}
			}
		}

		float view[16], proj[16], vp[16], eye[3];
		float aspect = fbh > 0 ? (float) fbw / fbh : 1.0f;
		camera_matrices(&cam, aspect, view, proj, vp, eye);

		// Bind virtual framebuffer for rendering
		if(use_vfb) {
			vfb_bind(&vfb);
			glEnable(GL_DEPTH_TEST);
		}

		renderer_begin_frame(ren, 0.15f, 0.15f, 0.2f, 1.0f);
		renderer_draw_instances(ren, scene.instances, scene.instance_count, vp, wireframe);

		// Render tendons as line segments
		if(m->ntendon > 0 && d->ten_wrapnum && d->ten_wrapadr && d->wrap_xpos) {
			int total_segments = 0;
			for(int t = 0; t < m->ntendon; ++t) {
				int n = d->ten_wrapnum[t];
				if(n >= 2) {
					total_segments += (n - 1);
				}
			}

			if(total_segments > 0) {
				size_t vertex_count = (size_t) total_segments * 2u;
				LineVertex *verts = (LineVertex *) malloc(sizeof(LineVertex) * vertex_count);
				if(verts) {
					const float color[4] = {1.0f, 0.9f, 0.2f, 1.0f};
					size_t v = 0;
					for(int t = 0; t < m->ntendon; ++t) {
						int n = d->ten_wrapnum[t];
						if(n < 2) {
							continue;
						}
						int adr = d->ten_wrapadr[t];
						for(int i = 0; i < n - 1; ++i) {
							const mjtNum *p0 = d->wrap_xpos + (adr + i) * 3;
							const mjtNum *p1 = d->wrap_xpos + (adr + i + 1) * 3;
							verts[v].position[0] = (float) p0[0];
							verts[v].position[1] = (float) p0[1];
							verts[v].position[2] = (float) p0[2];
							memcpy(verts[v].color, color, sizeof(color));
							v++;
							verts[v].position[0] = (float) p1[0];
							verts[v].position[1] = (float) p1[1];
							verts[v].position[2] = (float) p1[2];
							memcpy(verts[v].color, color, sizeof(color));
							v++;
						}
					}

					LineMesh *tendon_mesh = line_mesh_create(verts, vertex_count, LINE_TYPE_LIST);
					if(tendon_mesh) {
						LineInstance inst;
						memset(&inst, 0, sizeof(inst));
						inst.mesh = tendon_mesh;
						mat4_identity(inst.model);
						inst.override_color = 0;
						inst.mode = LINE_MODE_SOLID;
						inst.line_width = 2.0f;
						line_renderer_draw_lines(line_ren, &inst, 1, vp);
						line_mesh_destroy(tendon_mesh);
					}
					free(verts);
				}
			}
		}

		// Overlay text (shared for terminal + GUI)
		text_overlay_clear(&text);
		float ui_scale = terminal_mode ? 1.0f : 1.5f;
		text_overlay_printf_scaled(&text, 0.01f, 0.02f, 1.0f, 1.0f, 0.0f, 1.0f,
					   ui_scale, "r %.1fms", frame_time_ms);
		if(physics_rt_ratio > 0) {
			text_overlay_printf_scaled(&text, 0.01f, 0.06f, 1.0f, 1.0f, 0.8f, 1.0f,
						   ui_scale, "p %.1fms (%.0fx rt)", physics_time_ms,
						   physics_rt_ratio);
		}
		if(paused) {
			text_overlay_printf_scaled(&text, 0.85f, 0.02f, 1.0f, 0.3f, 0.3f, 1.0f,
						   ui_scale, "PAUSED");
		}
		if(speed_multiplier < 0.99f || speed_multiplier > 1.01f) {
			if(speed_multiplier >= 1.0f) {
				text_overlay_printf_scaled(&text, 0.01f, 0.10f, 1.0f, 0.3f, 0.3f, 1.0f,
							   ui_scale, "%.0fx speed", speed_multiplier);
			} else {
				int denom = (int) (1.0f / speed_multiplier + 0.5f);
				text_overlay_printf_scaled(&text, 0.01f, 0.10f, 1.0f, 0.3f, 0.3f, 1.0f,
							   ui_scale, "1/%dx speed", denom);
			}
		}

		if(use_vfb) {
			if(terminal_mode) {
				unsigned char *frame_rgb = vfb_read_pixels(&vfb);
				text_overlay_render_terminal(&text, &term);
				terminal_view_present(&term, frame_rgb, fbw, fbh);
				vfb_unbind();
			} else {
				if(term.braille_mode) {
					vfb_blit_to_screen_braille(&vfb, fbw, fbh);
				} else {
					vfb_blit_to_screen(&vfb, fbw, fbh);
				}
				text_overlay_render_gles(&text, fbw, fbh);
			}
		}

		glfwSwapBuffers(win);

		// Calculate actual render time (excluding sleep)
		double frame_end = glfwGetTime();
		frame_time_ms = (frame_end - frame_start) * 1000.0;
		last_time = frame_start; // For next frame's dt calculation

		// Frame rate limiting to ~30 fps
		double target_ms = 1000.0 / 30.0; // ~33.3ms for 30 fps
		if(frame_time_ms < target_ms) {
			double sleep_ms = target_ms - frame_time_ms;
			usleep((useconds_t) (sleep_ms * 1000.0)); // usleep takes microseconds
		}
	}

	input_shutdown(&g_input);
	text_overlay_shutdown(&text);
	if(terminal_mode) {
		terminal_view_shutdown(&term);
	}
	if(use_vfb) {
		vfb_destroy(&vfb);
	}
	renderer_destroy(ren);
	line_renderer_destroy(line_ren);
	mujoco_scene_free(&scene);
	// Free state history
	for(int i = 0; i < STATE_HISTORY_SIZE; i++) {
		mj_deleteData(state_history[i]);
	}
	free(state_history);
	glfwDestroyWindow(win);
	glfwTerminate();
	mj_deleteData(d);
	mj_deleteModel(m);
	return 0;
}