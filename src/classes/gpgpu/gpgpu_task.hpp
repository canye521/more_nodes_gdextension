#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/vector3i.hpp>

using namespace godot;


// 仅GDE侧使用
// GPUTask是纯数据结构,用于主线程向工作线程包装任务
class GPUTask : public RefCounted {

public:
    // 基本参数
    int64_t pipeline_id = 0;        // GPUPipeline 的 instance_id 更准确。
    String task_type = "compute";   // "compute" / "cleanup" / "readback_buffer"   任务类型
    String readback_buffer_name;   // 仅 "readback_buffer" 任务用

    // ---- create_display 任务专用 ----
    Array  display_handles;      // 主设备 3 张 display 的 handle（int64）
    String display_source_a;     // 源 pingpong 的 a 名字（如 "state_input"）
    String display_source_b;     // 源 pingpong 的 b 名字（如 "state_output"）
    int    display_width  = 0;
    int    display_height = 0;
    String display_format;       // "rgba32f"

    
    int iterations = 1;        //计算迭代次数
    Array pingpong_pairs;    // ping-pong 对（每个元素是 Dictionary：a_name / b_name）
    Array passes;


};