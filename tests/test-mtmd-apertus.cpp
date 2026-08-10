#include "mtmd-audio.h"
#include "mtmd-image.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

static void assert_size(clip_image_size actual, int expected_width, int expected_height) {
    assert(actual.width == expected_width);
    assert(actual.height == expected_height);
}

int main() {
    constexpr int spatial_factor = 16;
    constexpr int min_pixels = 65536;
    constexpr int max_pixels = 1960000;

    assert_size(mtmd_apertus_image_calc_size({429, 806}, spatial_factor, min_pixels, max_pixels), 432, 800);
    assert_size(mtmd_apertus_image_calc_size({1920, 1080}, spatial_factor, min_pixels, max_pixels), 1872, 1056);
    assert_size(mtmd_apertus_image_calc_size({1400, 1400}, spatial_factor, min_pixels, max_pixels), 1408, 1408);
    assert_size(mtmd_apertus_image_calc_size({4032, 3024}, spatial_factor, min_pixels, max_pixels), 1616, 1216);
    assert_size(mtmd_apertus_image_calc_size({1, 1}, spatial_factor, min_pixels, max_pixels), 256, 256);

    const std::vector<uint8_t> source = {
          0,  15,  30,  45,  60,  75,  90, 105, 120, 135, 150, 165,
        180, 195, 210, 225, 240,   0,  15,  30,  45,  60,  75,  90,
        105, 120, 135, 150, 165, 180, 195, 210, 225, 240,   0,  15,
    };
    const std::vector<uint8_t> expected = {
          0,   9,  30,  64,  79, 104, 133, 148, 162,
         81,  96,  81,  84,  98,  60,  86, 110, 130,
        203, 218, 156, 120, 135,  14,  37,  52,  81,
        154, 169, 153, 157, 177, 138, 159,  49,  70,
        107, 122, 143, 177, 201, 226, 246,  54,  68,
    };

    clip_image_u8 image;
    image.set_size({4, 3}, false);
    image.cpy_buf(source);
    const clip_image_f32 prepared = mtmd_apertus_image_prepare(image, {3, 5});
    assert_size(prepared.get_size(), 3, 5);

    const std::vector<float> & actual = prepared.get_ro_buf();
    assert(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const float normalized = ((float) expected[i] - 127.5f) / 127.5f;
        assert(actual[i] == normalized);
    }

    mtmd_audio_mel waveform;
    assert(!mtmd_apertus_audio_prepare(nullptr, 0, waveform));
    assert(!mtmd_apertus_audio_prepare(nullptr, 1, waveform));

    const std::vector<float> silence(600, 0.0f);
    assert(mtmd_apertus_audio_prepare(silence.data(), silence.size(), waveform));
    assert(waveform.n_len == 600);
    assert(waveform.n_len_org == 600);
    assert(waveform.n_mel == 1);
    assert(waveform.data == silence);

    const std::vector<float> samples = { 0.0f, 0.25f, -0.5f };
    assert(mtmd_apertus_audio_prepare(samples.data(), samples.size(), waveform));
    const float target_peak = std::pow(10.0f, -3.0f / 20.0f);
    assert(std::abs(waveform.data[0]) < 1e-7f);
    assert(std::abs(waveform.data[1] - 0.5f * target_peak) < 1e-7f);
    assert(std::abs(waveform.data[2] + target_peak) < 1e-7f);

    return 0;
}
