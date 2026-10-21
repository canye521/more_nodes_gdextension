#pragma once

#include "mn_common.hpp"         //LOG / JOIN / 断言宏

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/image.hpp>

using namespace godot;

//格式转换
// GLSL format 字符串 → RenderingDevice::DataFormat
// "rgba32f" → DATA_FORMAT_R32G32B32A32_SFLOAT
static RenderingDevice::DataFormat _glsl_format_to_rd(const String &format) {
    if (format == "rgba32f") return RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT;
    if (format == "rgba16f") return RenderingDevice::DATA_FORMAT_R16G16B16A16_SFLOAT;
    if (format == "rgba8")   return RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM;
    if (format == "rg32f")   return RenderingDevice::DATA_FORMAT_R32G32_SFLOAT;
    if (format == "rg16f")   return RenderingDevice::DATA_FORMAT_R16G16_SFLOAT;
    if (format == "rg8")     return RenderingDevice::DATA_FORMAT_R8G8_UNORM;
    if (format == "r32f")    return RenderingDevice::DATA_FORMAT_R32_SFLOAT;
    if (format == "r16f")    return RenderingDevice::DATA_FORMAT_R16_SFLOAT;
    if (format == "r8")      return RenderingDevice::DATA_FORMAT_R8_UNORM;
    // 默认 rgba32f
    LOG_W("[警告] 未知 GLSL format: ", format, "，用 rgba32f 为默认");
    return RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT;
}


// GLSL format 字符串 → Image::Format（读回时构造 Image 用）
static Image::Format _glsl_format_to_image(const String &format) {
    if (format == "rgba32f") return Image::FORMAT_RGBAF;
    if (format == "rgba16f") return Image::FORMAT_RGBAH;
    if (format == "rgba8")   return Image::FORMAT_RGBA8;
    if (format == "rg32f")   return Image::FORMAT_RGF;
    if (format == "rg16f")   return Image::FORMAT_RGH;
    if (format == "rg8")     return Image::FORMAT_RG8;
    if (format == "r32f")    return Image::FORMAT_RF;
    if (format == "r16f")    return Image::FORMAT_RH;
    if (format == "r8")      return Image::FORMAT_R8;
    return Image::FORMAT_RGBAF;
}