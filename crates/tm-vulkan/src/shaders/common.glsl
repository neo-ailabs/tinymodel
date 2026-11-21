// Shared GLSL 450 helpers and 32-element block dequantizers for all GGUF DTypes.
// Uses pure 32-bit `uint` SSBO reads (`unpackHalf2x16`), requiring ZERO optional
// 8-bit / 16-bit storage buffer extensions (portable across Mali, Adreno, V3DV, Desktop, MoltenVK).

#if defined(DT_Q4_0) || defined(DT_Q4_0X4) || defined(DT_Q4_1) || defined(DT_Q8_0) || defined(DT_Q5_K) || defined(DT_Q6_K) || defined(DT_F16) || defined(DT_BF16) || defined(DT_F32)

layout(set = 0, binding = 0, std430) readonly buffer WeightBuf {
    uint w_u32[];
};

#if !defined(NO_G_W32)
float g_w32[32];
#endif

uint w_load_u8(uint byte_off) {
    uint word = w_u32[byte_off >> 2u];
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

int w_load_s8(uint byte_off) {
    uint b = w_load_u8(byte_off);
    return (b >= 128u) ? (int(b) - 256) : int(b);
}

uint w_load_u16(uint byte_off) {
    uint b0 = w_load_u8(byte_off);
    uint b1 = w_load_u8(byte_off + 1u);
    return b0 | (b1 << 8u);
}

uint w_load_u32_unaligned(uint byte_off) {
    uint idx = byte_off >> 2u;
    uint rem = byte_off & 3u;
    uint lo = w_u32[idx];
    if (rem == 0u) {
        return lo;
    }
    uint sh = rem * 8u;
    // Note: when `rem in {1,2,3}`, `32 - sh in {24,16,8}` (strictly `< 32`).
    uint hi_sh = (4u - rem) * 8u;
    return (lo >> sh) | (w_u32[idx + 1u] << hi_sh);
}

float w_load_f16(uint byte_off) {
    return unpackHalf2x16(w_load_u16(byte_off)).x;
}

#if !defined(NO_G_W32)
void dequant_block32(uint row, uint row_bytes, uint cols_blocks32, uint blk32) {
#if defined(DT_Q4_0)
    uint boff = row * row_bytes + blk32 * 18u;
    float d = w_load_f16(boff);
    for (uint w = 0u; w < 4u; w++) {
        uint pack = w_load_u32_unaligned(boff + 2u + w * 4u);
        for (uint c = 0u; c < 4u; c++) {
            uint b = (pack >> (c * 8u)) & 0xFFu;
            uint j = w * 4u + c;
            g_w32[j] = float(int(b & 0xFu) - 8) * d;
            g_w32[j + 16u] = float(int(b >> 4u) - 8) * d;
        }
    }
#elif defined(DT_Q4_0X4)
    uint rg = row >> 2u;
    uint sub = row & 3u;
    uint sb_w = (rg * cols_blocks32 + blk32) * 18u;
    vec2 d_pair = unpackHalf2x16(w_u32[sb_w + (sub >> 1u)]);
    float d = ((sub & 1u) == 0u) ? d_pair.x : d_pair.y;
    uint half_base = sb_w + 2u + (sub >> 1u) * 8u + (sub & 1u) * 2u;
    uint words[4];
    words[0] = w_u32[half_base + 0u];
    words[1] = w_u32[half_base + 1u];
    words[2] = w_u32[half_base + 4u];
    words[3] = w_u32[half_base + 5u];
    for (uint w = 0u; w < 4u; w++) {
        uint pack = words[w];
        for (uint c = 0u; c < 4u; c++) {
            uint b = (pack >> (c * 8u)) & 0xFFu;
            uint j = w * 4u + c;
            g_w32[j] = float(int(b & 0xFu) - 8) * d;
            g_w32[j + 16u] = float(int(b >> 4u) - 8) * d;
        }
    }
#elif defined(DT_Q4_1)
    uint bw = (row * cols_blocks32 + blk32) * 5u;
    vec2 dm = unpackHalf2x16(w_u32[bw]);
    float d = dm.x;
    float m = dm.y;
    for (uint w = 0u; w < 4u; w++) {
        uint pack = w_u32[bw + 1u + w];
        for (uint c = 0u; c < 4u; c++) {
            uint b = (pack >> (c * 8u)) & 0xFFu;
            uint j = w * 4u + c;
            g_w32[j] = float(b & 0xFu) * d + m;
            g_w32[j + 16u] = float(b >> 4u) * d + m;
        }
    }
#elif defined(DT_Q8_0)
    uint boff = row * row_bytes + blk32 * 34u;
    float d = w_load_f16(boff);
    for (uint w = 0u; w < 8u; w++) {
        uint pack = w_load_u32_unaligned(boff + 2u + w * 4u);
        for (uint c = 0u; c < 4u; c++) {
            uint b = (pack >> (c * 8u)) & 0xFFu;
            int q = (b >= 128u) ? (int(b) - 256) : int(b);
            g_w32[w * 4u + c] = float(q) * d;
        }
    }
#elif defined(DT_Q5_K)
    uint ib = blk32 >> 3u;
    uint is = blk32 & 7u;
    uint boff = row * row_bytes + ib * 176u;
    float d = w_load_f16(boff);
    float dmin = w_load_f16(boff + 2u);
    uint sc;
    uint mn;
    if (is < 4u) {
        sc = w_load_u8(boff + 4u + is) & 63u;
        mn = w_load_u8(boff + 8u + is) & 63u;
    } else {
        uint b_hi = w_load_u8(boff + 8u + is);
        sc = (b_hi & 0xFu) | ((w_load_u8(boff + is) >> 6u) << 4u);
        mn = (b_hi >> 4u) | ((w_load_u8(boff + 4u + is) >> 6u) << 4u);
    }
    float d_sc = d * float(sc);
    float dm_m = dmin * float(mn);
    uint chunk = is >> 1u;
    uint hi_half = is & 1u;
    uint ql_w = (boff + 48u + chunk * 32u) >> 2u;
    uint qh_w = (boff + 16u) >> 2u;
    uint qh_sh = chunk * 2u + hi_half;
    for (uint w = 0u; w < 8u; w++) {
        uint ql_pack = w_u32[ql_w + w];
        uint qh_pack = w_u32[qh_w + w];
        for (uint c = 0u; c < 4u; c++) {
            uint ql_b = (ql_pack >> (c * 8u)) & 0xFFu;
            uint qh_b = (qh_pack >> (c * 8u)) & 0xFFu;
            uint nib = (hi_half == 0u) ? (ql_b & 0xFu) : (ql_b >> 4u);
            uint hb = ((qh_b >> qh_sh) & 1u) << 4u;
            g_w32[w * 4u + c] = d_sc * float(nib | hb) - dm_m;
        }
    }
#elif defined(DT_Q6_K)
    uint ib = blk32 >> 3u;
    uint sub8 = blk32 & 7u;
    uint boff = row * row_bytes + ib * 210u;
    float d = w_load_f16(boff + 208u);
    float sc0 = d * float(w_load_s8(boff + 192u + sub8 * 2u));
    float sc1 = d * float(w_load_s8(boff + 192u + sub8 * 2u + 1u));
    uint n128 = sub8 >> 2u;
    uint q_idx = sub8 & 3u;
    uint ql_base = boff + n128 * 64u + ((q_idx & 1u) * 32u);
    uint ql_sh = (q_idx >= 2u) ? 4u : 0u;
    uint qh_base = boff + 128u + n128 * 32u;
    uint qh_sh = q_idx * 2u;
    for (uint w = 0u; w < 8u; w++) {
        uint ql_pack = w_load_u32_unaligned(ql_base + w * 4u);
        uint qh_pack = w_load_u32_unaligned(qh_base + w * 4u);
        float scale = (w < 4u) ? sc0 : sc1;
        for (uint c = 0u; c < 4u; c++) {
            uint ql_b = (ql_pack >> (c * 8u)) & 0xFFu;
            uint qh_b = (qh_pack >> (c * 8u)) & 0xFFu;
            int q6 = int(((ql_b >> ql_sh) & 0xFu) | (((qh_b >> qh_sh) & 3u) << 4u)) - 32;
            g_w32[w * 4u + c] = scale * float(q6);
        }
    }
#elif defined(DT_F16)
    uint base_w = (row * cols_blocks32 + blk32) * 16u;
    for (uint w = 0u; w < 16u; w++) {
        vec2 p = unpackHalf2x16(w_u32[base_w + w]);
        g_w32[w * 2u] = p.x;
        g_w32[w * 2u + 1u] = p.y;
    }
#elif defined(DT_BF16)
    uint base_w = (row * cols_blocks32 + blk32) * 16u;
    for (uint w = 0u; w < 16u; w++) {
        uint u = w_u32[base_w + w];
        g_w32[w * 2u] = uintBitsToFloat((u & 0xFFFFu) << 16u);
        g_w32[w * 2u + 1u] = uintBitsToFloat(u & 0xFFFF0000u);
    }
#elif defined(DT_F32)
    uint base_w = (row * cols_blocks32 + blk32) * 32u;
    for (uint w = 0u; w < 32u; w++) {
        g_w32[w] = uintBitsToFloat(w_u32[base_w + w]);
    }
#endif
}
#endif

#endif
