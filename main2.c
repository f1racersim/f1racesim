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
double tilt_angle = 0.0; 
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
    int geom_id;
    double cl_alpha, cl0, cd0, k_induced, area;
} VLMAeroProperties;
static VLMAeroProperties v22_wing;

// --- KEYBOARD CALLBACK ---
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act != GLFW_PRESS && act != GLFW_REPEAT) return;
    if (key == GLFW_KEY_U) wind_speed -= 1.0;
    if (key == GLFW_KEY_J) wind_speed += 1.0;
    if (key == GLFW_KEY_R) { mj_resetData(m, d); tilt_angle = 0; }
    if (key == GLFW_KEY_UP) tilt_angle += 0.05;   
    if (key == GLFW_KEY_DOWN) tilt_angle -= 0.05; 
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
    int jnt_id = mj_name2id(m, mjOBJ_JOINT, "tilt_joint");
    if (jnt_id != -1) d->qpos[m->jnt_qposadr[jnt_id]] = tilt_angle;

    int gid = v22_wing.geom_id;
    mjtNum vel_world[6], local_vel[3], f_world[3], wind_vec[3] = {wind_speed, 0, 0};
    mj_objectVelocity(m, d, mjOBJ_GEOM, gid, vel_world, 0);
    mjtNum airspeed_world[3] = {vel_world[3] - wind_vec[0], vel_world[4], vel_world[5]};
    mju_mulMatTVec(local_vel, d->geom_xmat + 9 * gid, airspeed_world, 3, 3);

    double V = mju_norm3(local_vel) + 1e-6;
    double alpha = mju_clip(atan2(-local_vel[2], local_vel[0] + 1e-6), -0.5, 0.5); 
    double cl = v22_wing.cl0 + (v22_wing.cl_alpha * alpha);
    double cd = v22_wing.cd0 + (v22_wing.k_induced * cl * cl);
    double q_s = 0.5 * 1.225 * V * V * v22_wing.area;
    
    mjtNum f_local[3] = {-cd * q_s, 0, cl * q_s};
    mju_mulMatVec(f_world, d->geom_xmat + 9 * gid, f_local, 3, 3);
    mj_applyFT(m, d, f_world, NULL, d->geom_xpos + 3 * gid, m->geom_bodyid[gid], d->qfrc_passive);
    for(int i=3; i<6; i++) d->qfrc_passive[i] -= d->qvel[i] * 2.0;
}

int main(int argc, char** argv) {
    // --- 1. ROS3 & XBOX INITIALIZATION ---
    rose_node *node = rose_init((char)argc, argv, "v22_xbox_sim", NULL, NULL);
    rose_subscriber *sub = rose_create_sub(node, "/xbox/controller", -1, 1, NULL);
    mirage_msg *msg = mirage_create(1024, NULL);

    char fn_name[32];
    i64 nlen = 0, f_argc = 0;
    double axis_lx = 0, axis_ly = 0, trig_l = 0, axis_rx = 0;

    // --- 2. MUJOCO SETUP ---
    char* model_path = "testdata/v22_drone.xml";
    if (argc > 1) model_path = argv[1];

    if (!glfwInit()) return 1;
    char error[1000];
    m = mj_loadXML(model_path, NULL, error, 1000);
    if (!m) { printf("Error: %s\n", error); return 1; }
    d = mj_makeData(m);
    mjcb_passive = aerodynamics_callback;

    int wid = mj_name2id(m, mjOBJ_GEOM, "wing_geom");
    int bid = mj_name2id(m, mjOBJ_GEOM, "fuselage");
    if (wid == -1) wid = 0; 
    if (bid == -1) bid = 0;
    v22_wing = (VLMAeroProperties){wid, 4.8, 0.15, 0.02, 0.05, 0.3};

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
                mirage_read_f64(msg, &axis_ly); // Element 1
                mirage_read_f64(msg, &trig_l);  // Element 2
                mirage_read_f64(msg, &axis_rx); // Element 3

                // MAPPING: Stick Y controls Wing Tilt (inverted usually feels better for flight)
                tilt_angle = -axis_ly * 0.7; 
                
                // MAPPING: Stick X controls Wind Speed offset
                wind_speed = -15.0 + (axis_rx * 5.0);
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
                            particles[i].x = d->geom_xpos[3*bid] + 2.0;
                            particles[i].y = d->geom_xpos[3*bid+1] + dy;
                            particles[i].z = d->geom_xpos[3*bid+2] + dz;
                            particles[i].type = (fabs(dy) > 0.5) ? 1 : 0; 
                            break;
                        }
                    }
                }
            }
        }

        for(int i=0; i<MAX_PARTICLES; i++) {
            if(!particles[i].active) continue;
            float dx_b = particles[i].x - d->geom_xpos[3*bid];
            float dy_b = particles[i].y - d->geom_xpos[3*bid+1];
            float dz_b = particles[i].z - d->geom_xpos[3*bid+2];
            float dist_sq = dx_b*dx_b + dy_b*dy_b + dz_b*dz_b;
            float vy = 0, vz = 0;

            if (dist_sq < 0.5) {
                float repulsion = (0.5 - dist_sq) * 2.5;
                vy += (dy_b > 0 ? 1 : -1) * repulsion;
                vz += (dz_b > 0 ? 1 : -1) * repulsion;
            }

            float dx_w = particles[i].x - d->geom_xpos[3*wid];
            if (fabs(dx_w) < 0.3 && fabs(dy_b) < 0.7) {
                vz -= (dx_w < 0) ? 1.6 : -0.5; 
            }

            float prev[3] = {particles[i].x, particles[i].y, particles[i].z};
            particles[i].x += wind_speed * 0.016;
            particles[i].y += vy * 0.016;
            particles[i].z += vz * 0.016;
            particles[i].age += 0.016;
            if (particles[i].age > 1.2) particles[i].active = false;

            if (scn.ngeom < scn.maxgeom) {
                mjv_initGeom(&scn.geoms[scn.ngeom], mjGEOM_LINE, NULL, NULL, NULL, NULL);
                mjv_makeConnector(&scn.geoms[scn.ngeom], mjGEOM_LINE, 0.002, 
                                  prev[0], prev[1], prev[2], 
                                  particles[i].x, particles[i].y, particles[i].z);
                
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
        sprintf(hud, "Wind: %.1f | Tilt: %.2f | Controller: Connected", fabs(wind_speed), tilt_angle);
        mjr_overlay(mjFONT_BIG, mjGRID_TOPLEFT, viewport, hud, "Xbox: Left Stick (Tilt) | Right Stick (Wind)", &con);
        
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