#include "testing.h"

#include "mtmd-image.h"
#include "mtmd-internal.h"
#include "mtmd-helper.h"

#include <iostream>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)


//
// mtmd_image
//

MAKE_TEST(test_apertus_image_preprocessor) {
    const std::vector<std::pair<clip_image_size, clip_image_size>> sizes = {
        {{429, 806}, {432, 800}}, {{1920, 1080}, {1872, 1056}},
        {{1400, 1400}, {1408, 1408}}, {{4032, 3024}, {1616, 1216}}, {{1, 1}, {256, 256}},
    };
    for (const auto & entry : sizes) {
        const auto actual = mtmd_apertus_image_calc_size(entry.first, 16, 65536, 1960000);
        t.assert_equal(entry.second.width, actual.width);
        t.assert_equal(entry.second.height, actual.height);
    }
    for (const auto & args : std::vector<std::tuple<clip_image_size, int, int, int>> {
        {{0, 10}, 16, 65536, 1960000}, {{10, -1}, 16, 65536, 1960000},
        {{10, 10}, 0, 65536, 1960000}, {{10, 10}, 16, 0, 1960000},
        {{10, 10}, 16, 65536, 1},
        {{1, std::numeric_limits<int>::max()}, 16, 1, std::numeric_limits<int>::max()},
    }) {
        bool rejected = false;
        try {
            mtmd_apertus_image_calc_size(std::get<0>(args), std::get<1>(args), std::get<2>(args), std::get<3>(args));
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        t.assert_true("invalid dimensions rejected", rejected);
    }
    clip_image_u8 image;
    image.set_size({4, 3}, false);
    image.cpy_buf({
          0,  15,  30,  45,  60,  75,  90, 105, 120, 135, 150, 165,
        180, 195, 210, 225, 240,   0,  15,  30,  45,  60,  75,  90,
        105, 120, 135, 150, 165, 180, 195, 210, 225, 240,   0,  15,
    });
    // Torchvision antialiased uint8 bicubic, channels first, 4x3 -> 3x5.
    const std::vector<uint8_t> expected = {
          0,   9,  30,  64,  79, 104, 133, 148, 162,
         81,  96,  81,  84,  98,  60,  86, 110, 130,
        203, 218, 156, 120, 135,  14,  37,  52,  81,
        154, 169, 153, 157, 177, 138, 159,  49,  70,
        107, 122, 143, 177, 201, 226, 246,  54,  68,
    };
    const auto prepared = mtmd_apertus_image_prepare(image, {3, 5});
    const auto & pixels = prepared.get_ro_buf();
    if (t.assert_equal(expected.size(), pixels.size())) {
        for (size_t i = 0; i < pixels.size(); ++i) {
            t.assert_equal((float(expected[i]) - 127.5f) / 127.5f, pixels[i]);
        }
    }
}

MAKE_TEST(test_linear_media_v2) {
    std::vector<char> wire;
    const auto append = [&wire](auto value) {
        const auto * first = reinterpret_cast<const char *>(&value);
        wire.insert(wire.end(), first, first + sizeof(value));
    };
    append(uint64_t(2));
    append(uint32_t(MTMD_INPUT_CHUNK_TYPE_IMAGE));
    append(uint64_t(0));
    append(uint8_t(1));
    append(uint32_t(21)); // Full framed count, NORMAL positions, one image.
    append(uint32_t(1));
    append(uint32_t(0));
    append(uint32_t(0));
    append(uint32_t(1));
    append(uint64_t(0));
    append(uint8_t(0));
    append(uint64_t(1));
    append(uint8_t(0));
    append(uint8_t(0));
    append(int32_t(0)); // lead_pad
    append(int32_t(64));
    append(int32_t(64));
    append(uint8_t(0));
    mtmd::input_chunk_ptr chunk(mtmd_input_chunk_load(wire.data(), wire.size()));
    if (!t.assert_true("load v2 fixture", bool(chunk))) {
        return;
    }
    t.assert_equal(size_t(21), mtmd_input_chunk_get_n_tokens(chunk.get()));
    t.assert_equal(llama_pos(21), mtmd_input_chunk_get_n_pos(chunk.get()));
    const auto * image = mtmd_input_chunk_get_tokens_image(chunk.get());
    for (size_t i = 0; i < 21; ++i) {
        const auto pos = mtmd_image_tokens_get_decoder_pos(image, 5, i);
        t.assert_true("linear position", pos.t == 5 + uint32_t(i) && pos.x == pos.t && pos.y == pos.t && pos.z == pos.t);
    }
    size_t size = 0;
    t.assert_equal(0, mtmd_input_chunk_save(chunk.get(), nullptr, 0, &size));
    std::vector<char> saved(size);
    t.assert_equal(0, mtmd_input_chunk_save(chunk.get(), saved.data(), saved.size(), nullptr));
    t.assert_true("unchanged v2 bytes", saved == wire);
    const auto reject_field = [&](size_t offset, auto value) {
        auto invalid = wire;
        std::memcpy(invalid.data() + offset, &value, sizeof(value));
        mtmd::input_chunk_ptr loaded(mtmd_input_chunk_load(invalid.data(), invalid.size()));
        t.assert_true("invalid v2 field rejected", !loaded);
    };
    reject_field(0, uint64_t(1));
    reject_field(0, uint64_t(3));
    reject_field(21, uint32_t(0));
    reject_field(21, std::numeric_limits<uint32_t>::max());
    reject_field(37, uint32_t(0));
    reject_field(41, std::numeric_limits<uint64_t>::max());
    reject_field(50, std::numeric_limits<uint64_t>::max());
    reject_field(60, int32_t(-1));
    reject_field(64, int32_t(0));
    for (size_t length : {size_t(0), size_t(8), wire.size() - 1}) {
        mtmd::input_chunk_ptr loaded(mtmd_input_chunk_load(wire.data(), length));
        t.assert_true("truncated v2 chunk rejected", !loaded);
    }
    mtmd::input_chunk_ptr placeholder(mtmd_input_chunk_get_placeholder(chunk.get()));
    t.assert_equal(size_t(21), mtmd_input_chunk_get_n_tokens(placeholder.get()));

    std::vector<llama_token> tokens(21, 0);
    llama_pos next = -1;
    for (int batch : {0, -1}) {
        t.assert_equal(-1, mtmd_helper_decode_tokens_chunk(nullptr, nullptr, chunk.get(), tokens.data(), tokens.size(),
            0, 0, batch, &next, nullptr, nullptr));
    }
    t.assert_equal(-1, mtmd_helper_decode_tokens_chunk(nullptr, nullptr, chunk.get(), tokens.data(), tokens.size(),
        std::numeric_limits<llama_pos>::max() - 20, 0, 8, &next, nullptr, nullptr));
    t.assert_equal(-1, mtmd_helper_decode_tokens_chunk(nullptr, nullptr, chunk.get(), tokens.data(), tokens.size() - 1,
        0, 0, 8, &next, nullptr, nullptr));
    t.assert_equal(llama_pos(-1), next);
}

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);

    // { image size, expected tiling }
    const std::vector<std::pair<clip_image_size, bool>> cases = {
        { {  704, 704 }, false },
        // 720 / (patch_size * n_merge) is exactly 22.5, so this only matches HF
        // if round_by_factor rounds half to even (22) instead of away from zero (23)
        { {  720, 720 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & [size, expected] : cases) {
        const bool actual = mtmd_image_preprocessor_lfm2::should_tile(hparams, size);

        t.assert_equal(
            "tiling for " + std::to_string(size.width) + "x" + std::to_string(size.height),
            std::string(expected ? "tiled" : "single"),
            std::string(actual   ? "tiled" : "single"));
    }
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_internal_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
