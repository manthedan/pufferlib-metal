#pragma once

// CPU environment for the Metal learner. Physics and reset/reward logic remain
// in the upstream sources; each worker invokes their kernels for one Env.
#define RA_CPU 1
#include "robot_arm_cuda.cuh"

static RaRenderHost g_ra_cpu_render;

static int ra_cpu_flag(Dict* kwargs, const char* key) {
    DictItem* item = dict_find(kwargs, key);
    return item != NULL && item->value != 0.0;
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    // The native CPU vector passes a zero-based index; CUDA seeds i + 1.
    env->world.state.rng = env->rng + 1u;
    env->world.state.no_timeout = ra_cpu_flag(kwargs, "no_timeout");
    env->world.state.stack_mode = ra_cpu_flag(kwargs, "stack");
    env->world.state.basketball_mode = ra_cpu_flag(kwargs, "basketball");
    assert(!(env->world.state.stack_mode && env->world.state.basketball_mode));
    ra_reset(&env->world.state, 1);
    ra_rbrst(&env->world.rigid, ra_topo(&env->world.state));
    DictItem* model = dict_find(kwargs, "model_glb");
    g_ra_cpu_render.model_glb = model && model->str && model->str[0]
        && strcmp(model->str, "None") != 0 ? model->str
        : "resources/robot_arm/franka_panda.glb";
    g_ra_cpu_render.camera_distance = env->world.state.basketball_mode ? 2.35f : 1.55f;
    g_ra_cpu_render.camera_yaw = 0.78f;
    g_ra_cpu_render.camera_pitch = 0.48f;
}

void puf_reset(Env* env) {
    Agent* agent = &env->agents[0];
    ra_kinit(env, agent->observations, agent->rewards, agent->terminals, 1);
}

void puf_step(Env* env) {
    Agent* agent = &env->agents[0];
    ra_kbegin(env, 0, 1, agent->actions);
    ra_kphys(env, 0, 1);
    ra_kfin(env, 0, 1, agent->observations, agent->rewards, agent->terminals);
}

void puf_close(Env*) {
    ra_rclose(&g_ra_cpu_render);
}

void puf_render(Env* env) {
    RaState* state = &env->world.state;
    RaPose links[RA_LINKS];
    ra_fk(state->q, state->gripper_width, links, NULL, NULL, &state->end_effector);
    ra_draw(&g_ra_cpu_render, state, links);
    if (g_ra_cpu_render.reset_requested) {
        g_ra_cpu_render.reset_requested = 0;
        ra_reset(state, 1);
        ra_rbrst(&env->world.rigid, ra_topo(state));
        puf_reset(env);
    }
}
