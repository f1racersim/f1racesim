#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

// ROS3 / Mirage Headers
#include "ros3.h"

// --- GLOBALS ---
mjModel* m = NULL;
mjData* d = NULL;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

double wind_speed = -15.0; 
bool track_mode = true;
bool button_left = false, button_right = false;
double lastx = 0, lasty = 0;

// Particle System
#define MAX_PARTICLES 3000 
typedef struct {
    float x, y, z;
    bool active;
    float age;
    int type; 
} Particle;
Particle particles[MAX_PARTICLES];

typedef struct {
    int joint_id;
    int geom_id;
    double angle;
    double cl_alpha, cl0, cd0, k_induced, area;
} AeroFlap;

enum {
    LEFT_FLAP = 0,
    RIGHT_FLAP = 1,
    NUM_FLAPS = 2
};

static AeroFlap flaps[NUM_FLAPS];
static int flap_count = 0;
static int fuselage_geom_id = -1;

static double clamp_flap_angle(double angle) {
    return mju_clip(angle, -1.2, 1.2);
}

static double normalize_trigger(double axis_value) {
    return 0.5 * (mju_clip(axis_value, -1.0, 1.0) + 1.0);
}

static int lookup_id(const mjModel* model, int objtype, const char* name) {
    return mj_name2id(model, objtype, name);
}

static void apply_flap_aero(const mjModel* model, mjData* data, AeroFlap* flap) {
    mjtNum vel_world[6], local_vel[3], f_world[3];
    mjtNum wind_vec[3] = {(mjtNum) wind_speed, 0, 0};

    if (flap->joint_id != -1) {
        data->qpos[model->jnt_qposadr[flap->joint_id]] = clamp_flap_angle(flap->angle);
    }

    if (flap->geom_id == -1) {
        return;
    }

    mj_objectVelocity(model, data, mjOBJ_GEOM, flap->geom_id, vel_world, 0);
    mjtNum airspeed_world[3] = {vel_world[3] - wind_vec[0], vel_world[4], vel_world[5]};
    mju_mulMatTVec(local_vel, data->geom_xmat + 9 * flap->geom_id, airspeed_world, 3, 3);

    double speed = mju_norm3(local_vel) + 1e-6;
    double alpha = mju_clip(atan2(-local_vel[2], local_vel[0] + 1e-6), -0.8, 0.8);
    double cl = flap->cl0 + (flap->cl_alpha * alpha);
    double cd = flap->cd0 + (flap->k_induced * cl * cl);
    double q_s = 0.5 * 1.225 * speed * speed * flap->area;

    mjtNum f_local[3] = {(mjtNum)(-cd * q_s), 0, (mjtNum)(cl * q_s)};
    mju_mulMatVec(f_world, data->geom_xmat + 9 * flap->geom_id, f_local, 3, 3);
    mj_applyFT(model, data, f_world, NULL, data->geom_xpos + 3 * flap->geom_id,
               model->geom_bodyid[flap->geom_id], data->qfrc_passive);
}

// --- KEYBOARD CALLBACK ---
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act != GLFW_PRESS && act != GLFW_REPEAT) return;
    if (key == GLFW_KEY_U) wind_speed -= 1.0;
    if (key == GLFW_KEY_J) wind_speed += 1.0;
    if (key == GLFW_KEY_R) {
        mj_resetData(m, d);
        for (int flap_idx = 0; flap_idx < flap_count; flap_idx++) {
            flaps[flap_idx].angle = 0.0;
        }
    }
    if (key == GLFW_KEY_UP) {
        for (int flap_idx = 0; flap_idx < flap_count; flap_idx++) {
            flaps[flap_idx].angle = clamp_flap_angle(flaps[flap_idx].angle + 0.05);
        }
    }
    if (key == GLFW_KEY_DOWN) {
        for (int flap_idx = 0; flap_idx < flap_count; flap_idx++) {
            flaps[flap_idx].angle = clamp_flap_angle(flaps[flap_idx].angle - 0.05);
        }
    }
    if (flap_count > 0 && key == GLFW_KEY_A) flaps[LEFT_FLAP].angle = clamp_flap_angle(flaps[LEFT_FLAP].angle + 0.05);
    if (flap_count > 0 && key == GLFW_KEY_Z) flaps[LEFT_FLAP].angle = clamp_flap_angle(flaps[LEFT_FLAP].angle - 0.05);
    if (flap_count > 1 && key == GLFW_KEY_K) flaps[RIGHT_FLAP].angle = clamp_flap_angle(flaps[RIGHT_FLAP].angle + 0.05);
    if (flap_count > 1 && key == GLFW_KEY_M) flaps[RIGHT_FLAP].angle = clamp_flap_angle(flaps[RIGHT_FLAP].angle - 0.05);
    if (key == GLFW_KEY_C) {
        track_mode = !track_mode;
        cam.type = track_mode ? mjCAMERA_TRACKING : mjCAMERA_FREE;
        if (track_mode) cam.trackbodyid = mj_name2id(m, mjOBJ_BODY, "v22_body");
    }
}

// --- MOUSE CALLBACKS ---
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &lastx, &lasty);
}

void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    if (!button_left && !button_right) return;
    double dx = xpos - lastx; double dy = ypos - lasty;
    lastx = xpos; lasty = ypos;
    int width, height; glfwGetWindowSize(window, &width, &height);
    mjtMouse action = button_right ? mjMOUSE_ZOOM : (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) ? mjMOUSE_MOVE_V : mjMOUSE_ROTATE_V);
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}

// --- AERODYNAMICS CALLBACK ---
void aerodynamics_callback(const mjModel* m, mjData* d) {
    for (int flap_idx = 0; flap_idx < flap_count; flap_idx++) {
        apply_flap_aero(m, d, &flaps[flap_idx]);
    }

    for (int i = 3; i < 6 && i < m->nv; i++) {
        d->qfrc_passive[i] -= d->qvel[i] * 2.0;
    }
}

int main(int argc, char** argv) {
    // --- 1. ROS3 & XBOX INITIALIZATION ---
    rose_node *node = rose_init((char)argc, argv, "v22_xbox_sim", NULL, NULL);
    rose_subscriber *sub = rose_create_sub(node, "/xbox/controller", -1, 1, NULL);
    mirage_msg *msg = mirage_create(1024, NULL);

    char fn_name[32];
    i64 nlen = 0, f_argc = 0;
    double axis_lx = 0, axis_ly = 0, trig_l = -1.0, axis_rx = 0, axis_ry = 0, trig_r = -1.0;

    // --- 2. MUJOCO SETUP ---
    char* model_path = "testdata/v22_drone.xml";
    if (argc > 1) model_path = argv[1];

    if (!glfwInit()) return 1;
    char error[1000];
    m = mj_loadXML(model_path, NULL, error, 1000);
    if (!m) { printf("Error: %s\n", error); return 1; }
    d = mj_makeData(m);
    mjcb_passive = aerodynamics_callback;

    memset(flaps, 0, sizeof(flaps));
    flap_count = 0;

    int left_joint_id = lookup_id(m, mjOBJ_JOINT, "left_flap_joint");
    int left_geom_id = lookup_id(m, mjOBJ_GEOM, "left_wing_geom");
    int right_joint_id = lookup_id(m, mjOBJ_JOINT, "right_flap_joint");
    int right_geom_id = lookup_id(m, mjOBJ_GEOM, "right_wing_geom");

    if (left_joint_id != -1 && left_geom_id != -1 &&
        right_joint_id != -1 && right_geom_id != -1) {
        flap_count = 2;
        flaps[LEFT_FLAP] = (AeroFlap){
            .joint_id = left_joint_id,
            .geom_id = left_geom_id,
            .angle = 0.0,
            .cl_alpha = 4.8,
            .cl0 = 0.15,
            .cd0 = 0.02,
            .k_induced = 0.05,
            .area = 0.15
        };
        flaps[RIGHT_FLAP] = (AeroFlap){
            .joint_id = right_joint_id,
            .geom_id = right_geom_id,
            .angle = 0.0,
            .cl_alpha = 4.8,
            .cl0 = 0.15,
            .cd0 = 0.02,
            .k_induced = 0.05,
            .area = 0.15
        };
    } else {
        int single_joint_id = lookup_id(m, mjOBJ_JOINT, "tilt_joint");
        int single_geom_id = lookup_id(m, mjOBJ_GEOM, "wing_geom");
        if (single_joint_id != -1 && single_geom_id != -1) {
            flap_count = 1;
            flaps[LEFT_FLAP] = (AeroFlap){
                .joint_id = single_joint_id,
                .geom_id = single_geom_id,
                .angle = 0.0,
                .cl_alpha = 4.8,
                .cl0 = 0.15,
                .cd0 = 0.02,
                .k_induced = 0.05,
                .area = 0.3
            };
        }
    }

    fuselage_geom_id = lookup_id(m, mjOBJ_GEOM, "fuselage");

    GLFWwindow* window = glfwCreateWindow(1200, 900, "V22 Volumetric VLM - Xbox Integrated", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyboard);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetCursorPosCallback(window, mouse_move);

    mjv_defaultCamera(&cam); mjv_defaultOption(&opt); mjv_defaultScene(&scn); mjr_defaultContext(&con);
    cam.type = mjCAMERA_TRACKING; cam.trackbodyid = mj_name2id(m, mjOBJ_BODY, "v22_body");
    mjv_makeScene(m, &scn, 10000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // --- 3. MAIN LOOP ---
    while (!glfwWindowShouldClose(window) && rose_ok(node)) {
        
        // POLL XBOX CONTROLLER
        if (rose_read(sub, msg) >= 0) {
            mirage_read_start(msg);
            mirage_read_fn(msg, fn_name, &nlen, sizeof(fn_name), &f_argc);
            
            if (f_argc >= 4) {
                mirage_read_f64(msg, &axis_lx); // Element 0
                mirage_read_f64(msg, &axis_ly); // Element 1: left stick Y
                mirage_read_f64(msg, &trig_l);  // Element 2: left trigger
                mirage_read_f64(msg, &axis_rx); // Element 3: right stick X
                axis_ry = axis_rx;
                if (f_argc >= 5) mirage_read_f64(msg, &axis_ry); // Element 4: right stick Y
                if (f_argc >= 6) mirage_read_f64(msg, &trig_r);  // Element 5: right trigger

                if (flap_count >= 2) {
                    flaps[LEFT_FLAP].angle = clamp_flap_angle(-axis_ly * 0.9);
                    flaps[RIGHT_FLAP].angle = clamp_flap_angle(-axis_ry * 0.9);
                } else if (flap_count == 1) {
                    flaps[LEFT_FLAP].angle = clamp_flap_angle(-axis_ly * 0.9);
                }
                wind_speed = -15.0 + ((normalize_trigger(trig_r) - normalize_trigger(trig_l)) * 5.0);
            }
        }

        mjtNum simstart = d->time;
        while (d->time - simstart < 1.0/60.0) mj_step(m, d);
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

        // --- PARTICLE LOGIC ---
        static int timer = 0;
        if (timer++ > 1) {
            timer = 0;
            for(double dy = -0.8; dy <= 0.8; dy += 0.12) {
                for(double dz = -0.4; dz <= 0.4; dz += 0.12) {
                    for(int i=0; i<MAX_PARTICLES; i++) {
                        if(!particles[i].active) {
                            particles[i].active = true; particles[i].age = 0;
                            if (fuselage_geom_id != -1) {
                                particles[i].x = d->geom_xpos[3*fuselage_geom_id] + 2.0;
                                particles[i].y = d->geom_xpos[3*fuselage_geom_id+1] + dy;
                                particles[i].z = d->geom_xpos[3*fuselage_geom_id+2] + dz;
                            } else {
                                particles[i].x = 2.0;
                                particles[i].y = dy;
                                particles[i].z = dz;
                            }
                            particles[i].type = (fabs(dy) > 0.5) ? 1 : 0; 
                            break;
                        }
                    }
                }
            }
        }

        for(int i=0; i<MAX_PARTICLES; i++) {
            if(!particles[i].active) continue;
            float body_x = fuselage_geom_id != -1 ? d->geom_xpos[3*fuselage_geom_id] : 0.0f;
            float body_y = fuselage_geom_id != -1 ? d->geom_xpos[3*fuselage_geom_id+1] : 0.0f;
            float body_z = fuselage_geom_id != -1 ? d->geom_xpos[3*fuselage_geom_id+2] : 0.0f;
            float dx_b = particles[i].x - body_x;
            float dy_b = particles[i].y - body_y;
            float dz_b = particles[i].z - body_z;
            float dist_sq = dx_b*dx_b + dy_b*dy_b + dz_b*dz_b;
            float vy = 0, vz = 0;

            if (dist_sq < 0.5) {
                float repulsion = (0.5 - dist_sq) * 2.5;
                vy += (dy_b > 0 ? 1 : -1) * repulsion;
                vz += (dz_b > 0 ? 1 : -1) * repulsion;
            }

            for (int flap_idx = 0; flap_idx < flap_count; flap_idx++) {
                int flap_geom_id = flaps[flap_idx].geom_id;
                float dx_w = particles[i].x - d->geom_xpos[3*flap_geom_id];
                float dy_w = particles[i].y - d->geom_xpos[3*flap_geom_id+1];

                if (fabs(dx_w) < 0.3f && fabs(dy_w) < 0.35f) {
                    vz -= (dx_w < 0 ? 1.2f : -0.3f);
                    vz -= (float)(flaps[flap_idx].angle * 1.5);
                }
            }

            float prev[3] = {particles[i].x, particles[i].y, particles[i].z};
            particles[i].x += wind_speed * 0.016;
            particles[i].y += vy * 0.016;
            particles[i].z += vz * 0.016;
            particles[i].age += 0.016;
            if (particles[i].age > 1.2) particles[i].active = false;

            if (scn.ngeom < scn.maxgeom) {
                mjtNum from[3] = {prev[0], prev[1], prev[2]};
                mjtNum to[3] = {particles[i].x, particles[i].y, particles[i].z};
                mjv_initGeom(&scn.geoms[scn.ngeom], mjGEOM_LINE, NULL, NULL, NULL, NULL);
                mjv_connector(&scn.geoms[scn.ngeom], mjGEOM_LINE, 1.0, from, to);
                
                float current_v = sqrt(pow(wind_speed, 2) + pow(vy, 2) + pow(vz, 2));
                float speed_diff = (current_v - fabs(wind_speed)) / 3.0; 
                float speed_factor = mju_clip(speed_diff, -1.0, 1.0);

                scn.geoms[scn.ngeom].rgba[0] = 0.5 + (0.5 * speed_factor);
                scn.geoms[scn.ngeom].rgba[1] = 0.8 - (0.5 * fabs(speed_factor));
                scn.geoms[scn.ngeom].rgba[2] = 0.5 - (0.5 * speed_factor);
                scn.geoms[scn.ngeom].rgba[3] = 1.0 - (particles[i].age / 1.2);
                scn.ngeom++;
            }
        }

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        mjr_render(viewport, &scn, &con);
        
        char hud[128]; 
        if (flap_count >= 2) {
            sprintf(hud, "Wind: %.1f | Left flap: %.2f | Right flap: %.2f",
                    fabs(wind_speed), flaps[LEFT_FLAP].angle, flaps[RIGHT_FLAP].angle);
            mjr_overlay(mjFONT_BIG, mjGRID_TOPLEFT, viewport, hud,
                        "Xbox: Left Y -> left flap | Right Y -> right flap | Triggers -> wind", &con);
        } else if (flap_count == 1) {
            sprintf(hud, "Wind: %.1f | Flap: %.2f",
                    fabs(wind_speed), flaps[LEFT_FLAP].angle);
            mjr_overlay(mjFONT_BIG, mjGRID_TOPLEFT, viewport, hud,
                        "Xbox: Left Y -> flap | Triggers -> wind", &con);
        } else {
            sprintf(hud, "Wind: %.1f | No flap surfaces detected", fabs(wind_speed));
            mjr_overlay(mjFONT_BIG, mjGRID_TOPLEFT, viewport, hud,
                        "Triggers -> wind", &con);
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // --- 4. CLEANUP ---
    mirage_destroy(&msg, NULL);
    mj_deleteData(d);
    mj_deleteModel(m);
    mjr_freeContext(&con);
    mjv_freeScene(&scn);
    glfwTerminate();
    
    return 0;
}
