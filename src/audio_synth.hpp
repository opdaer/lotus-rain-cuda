#pragma once

#include "common.cuh"
#include <string>
#include <vector>

struct AudioDiagnostics {
    float max_abs_sample = 0.0f;
    float rms_0_4s       = 0.0f;
    float rms_4_14s      = 0.0f;
    float rms_14_20s     = 0.0f;
    int   pond_events    = 0;
    int   pearl_events   = 0;
    int   leaf_events    = 0;
    bool  has_nan        = false;
};

AudioDiagnostics synthesize_coupled_pond_audio_wav(
    const std::vector<GpuImpactEvent>& events,
    const LotusLeafState leaves[NUM_LEAVES],
    const std::string& wav_output_path,
    float duration_sec = DURATION_SEC,
    int sample_rate = 48000
);
