#pragma once
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <vector>

#include "mn_common.hpp" //LOG / JOIN / 断言宏
#include <godot_cpp/classes/texture2drd.hpp>
using namespace godot;

// PassData —— 单个 pass 的解析结果
struct PassData
{
    // 解析结果
    String shader_path;
    bool glsl_parsed = false;                // 管线GLSL是否已经解析
    Dictionary binding_map;                  // name → {set, binding, type, format, access, ...}
    Vector3i local_size = Vector3i(1, 1, 1); // 从 GLSL 解析出的工作组大小（local_size_x/y/z）
    Array push_const_decls;                  // push_const_decls 字段列表：[{type, name}, ...]
    int push_const_size = 0;                 // push_constant 总字节数（GLSL 解析后算出）
    bool has_push_const = false;             // GLSL 里是否有 push_constant
    Array spec_const_decls;

    // ---- 用户数据（set_* 写）----
    PackedByteArray push_const_bytes;
    Dictionary spec_constants;
    Vector3i dispatch_groups = Vector3i(0, 0, 0); // (0,0,0) 或任一分量≤0 = 自动计算
};

class GPUPipeline : public RefCounted
{
    GDCLASS(GPUPipeline, RefCounted)
protected:
    static void _bind_methods();

public:
    // 构造函数 / 析构函数
    GPUPipeline();
    ~GPUPipeline();

    // 提交任务清单
    void submit_task();

    void set_iterations(int n);                                          // 设置迭代次数
    void set_dispatch_groups(const Vector3i &groups, int pass_idx = -1); // 设置派发工作组数
    // 管线的着色器路径
    void add_pass(const String &shader_path);
    int get_pass_count() const;                      // pass 数量
    String get_pass_shader_path(int pass_idx) const; // 第 i 个 pass 的 shader 路径
    PackedStringArray get_all_shader_paths() const;  // 全部 pass 的 shader 路径

    // 数据绑定
    void set_push_const(const Variant &data, int pass_idx = 0);          // push constant 数据           每次调用 dispatch 都会上传
    void set_spec_const(int id, const Variant &value, int pass_idx = 0); // 设置特化常量,一个shader执行   创建管线时确定

    // 纹理数据
    void set_image(const String &name, const Variant &data);
    Ref<Image> get_image(const String &name);

    // 跨管线传 RID
    RID get_texture_rid(const String &name);
    void set_texture_rid(const String &name, RID rid);

    // 获取纹理
    Ref<Texture2DRD> get_latest_t2drd(const String &name);

    // 统一缓冲数据
    void set_uniform_buffer(const String &name, const Variant &data);

    // 存储缓冲数据
    void set_storage_buffer(const String &name, const Variant &data);
    void request_storage_buffer(const String &name);

    // ping-pong 双缓冲
    //   a_name / b_name 是 GLSL 里两个 image2D 的名字
    //   每次 iteration 会交替"读 a 写 b" / "读 b 写 a"
    void set_pingpong(const String &a_name, const String &b_name);
    Variant get_pingpong_output(const String &name);

    void set_pair_flip_counts(const Dictionary &counts); // 设置返回的乒乓相位 GPGPU设置
    Dictionary get_pair_flip_counts() const;

    // link：让 pass_a 里的 name_a 和 pass_b 里的 name_b 指向同一张纹理
    void link(int pass_a, const String &name_a,
              int pass_b, const String &name_b);

    // 手动标脏
    void mark_dirty_storage_buffer(const String &name);
    void mark_dirty_uniform_buffer(const String &name);

    // Display 三缓冲
    //   主线程建 3 张 display，worker compute 后 copy，主线程收到 show_idx 切换
    //   source_name 必须是已绑定的 pingpong 对的一端（a 或 b）
    void create_display(const String &source_name, int width, int height);
    Ref<Texture2DRD> get_display_texture();
    void apply_display_flip(int show_idx); // 仅 GPGPU::_on_process_frame 调用（不 bind GDS）

    // 生成绑定的 GDScript 模板代码（返回字符串，用户可复制）
    String generate_bindings_code() const;

private:                                                   // 私有
    std::vector<PassData> _passes;                         // 每个元素是 Dictionary（pass 数据）
    void _parse_glsl(int pass_idx);                        // 解析 GLSL 着色器，提取绑定信息
    int _find_pass_with_binding(const String &name) const; // 找到包含该绑定的 pass，返回 -1 表示没有
    void _validate_bindings() const;                       // 检查每个 pass 声明的绑定是否都 set 过

    // 用户数据
    Dictionary _init_images;     // name → Ref<Image>，仅 Image 路径，用于首次上传 init_data
    Dictionary _readback_images; // name → Ref<Image>，CPU 侧 Image 缓存（未脏返回值 / GPU 读回结果）

    Dictionary _storage_buffers; // name → PackedFloat32Array
    Dictionary _uniform_buffers; // name → {type, data}

    // 共享纹理
    Dictionary _main_texture_ids; // name → int64 tex_id（注册表里的编号）

    // 脏标记(buffer用 打包任务时判断哪些要传)
    Dictionary _init_sent;             // tex_id → true（已发过 init_data）
    Dictionary _dirty_storage_buffers; // name → true
    Dictionary _dirty_uniform_buffers; // name → true

    // 读回
    //  待读回标记（GPU 侧：dispatch 后结果比 CPU 缓存新，get 时才读）
    Dictionary _pending_readback; // name → true

    // 计算参数
    int _iterations = 1; // 迭代次数

    // 乒乓
    Array _pingpong_a_names;      // 每对的 a 名字
    Array _pingpong_b_names;      // 每对的 b 名字
    Dictionary _pair_flip_counts; // worker 回传的最终相位

    Dictionary _link_map; // name → other_name（双向映射，1:1）

    // ---- display 三缓冲 ----
    static constexpr int DISPLAY_COUNT = 3;
    int64_t _display_tex_ids[DISPLAY_COUNT] = {-1, -1, -1};
    Ref<Texture2DRD> _display_t2drd;
    bool _display_enabled = false;
    String _display_source_a;
    String _display_source_b;
    int _display_current_show = -1;

    // 内部：创建 RD 纹理 + 注册到注册表（ref_count=1）
    int64_t _create_rd_texture(int w, int h, const String &fmt, uint64_t extra_usage, RID *out_rid = nullptr);

    String _find_bound_name(const String &name, const Dictionary &table) const; // 返回真实有绑定的名字，找不到返回空串
    // 解析"最新那张"的真实绑定名（处理 link 别名 + pingpong 翻转）
    //   返回空串 → name 未绑定
    //   is_pingpong_out（可选）→ 告知结果是否来自 pingpong 对
    String _resolve_latest_name(const String &name, bool *is_pingpong_out = nullptr) const;
};