#include "gpu_pipeline.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/texture2drd.hpp>

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/reg_ex.hpp>
#include <godot_cpp/classes/reg_ex_match.hpp>

#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/classes/os.hpp>

#include "gpgpu.hpp"
#include "gpgpu_common.hpp"

#pragma region 辅助方法
// 仅本文件使用

// GLSL 类型 → std140 对齐
static int _glsl_align_std140(const String &t)
{
    if (t == "double")
        return 8;
    if (t == "vec2" || t == "ivec2" || t == "uvec2")
        return 8;
    if (t == "vec3" || t == "ivec3" || t == "uvec3")
        return 16;
    if (t == "vec4" || t == "ivec4" || t == "uvec4")
        return 16;
    if (t == "mat2")
        return 8;
    if (t == "mat3")
        return 16;
    if (t == "mat4")
        return 16;
    return 4;
}

// GLSL 类型 → std140 大小
static int _glsl_size_std140(const String &t)
{
    if (t == "double")
        return 8;
    if (t == "vec2" || t == "ivec2" || t == "uvec2")
        return 8;
    if (t == "vec3" || t == "ivec3" || t == "uvec3")
        return 12;
    if (t == "vec4" || t == "ivec4" || t == "uvec4")
        return 16;
    if (t == "mat2")
        return 16;
    if (t == "mat3")
        return 36;
    if (t == "mat4")
        return 64;
    return 4;
}

// GLSL 类型 → std430 对齐
static int _glsl_align_std430(const String &t)
{
    if (t == "double")
        return 8;
    if (t == "vec2" || t == "ivec2" || t == "uvec2")
        return 8;
    if (t == "vec3" || t == "ivec3" || t == "uvec3")
        return 16;
    if (t == "vec4" || t == "ivec4" || t == "uvec4")
        return 16;
    if (t == "mat2")
        return 8;
    if (t == "mat3")
        return 16;
    if (t == "mat4")
        return 16;
    return 4; // int / uint / float / bool
}

// GLSL 类型 → std430 大小
static int _glsl_size_std430(const String &t)
{
    if (t == "double")
        return 8;
    if (t == "vec2" || t == "ivec2" || t == "uvec2")
        return 8;
    if (t == "vec3" || t == "ivec3" || t == "uvec3")
        return 12;
    if (t == "vec4" || t == "ivec4" || t == "uvec4")
        return 16;
    if (t == "mat2")
        return 16;
    if (t == "mat3")
        return 36;
    if (t == "mat4")
        return 64;
    return 4;
}

// 向上对齐
static int _align_up(int value, int align)
{
    return (value + align - 1) / align * align;
}

// 在 _spec_const_decls 里找 id 对应的声明，找不到返回空字典
static Dictionary _find_spec_decl(const Array &decls, int id)
{
    for (int i = 0; i < decls.size(); i++)
    {
        Dictionary d = decls[i];
        if ((int)d["id"] == id)
            return d;
    }
    return Dictionary();
}

// 检查 name 是否在 _binding_map 里，且类型匹配
// 返回：0=OK，1=名字不存在，2=类型不匹配
static int _check_binding(const Dictionary &binding_map,
                          const String &name, const String &expect_type)
{
    if (!binding_map.has(name))
        return 1;
    Dictionary info = binding_map[name];
    if ((String)info["type"] != expect_type)
        return 2;
    return 0;
}

#pragma endregion

#pragma region GLSL 解析

// GLSL 解析器
//   读 .glsl 文本，用正则扫出所有 layout 绑定
//   结果存进 _binding_map：名字 → {set, binding, type, format, access}
//   每个 pipeline 只解析一次（_glsl_parsed 标记）
void GPUPipeline::_parse_glsl(int pass_idx)
{
    PassData &pd = _passes[pass_idx];
    // 标注检查,防止重复解析
    if (pd.glsl_parsed)
        return;
    pd.glsl_parsed = true;

    // 读文件内容
    Ref<FileAccess> f = FileAccess::open(pd.shader_path, FileAccess::READ);
    if (f.is_null())
    {
        LOG_E("[GPUPipeline] 无法打开 GLSL: ", pd.shader_path);
        return;
    }
    String text = f->get_as_text();
    f->close();

    // 创建正则
    Ref<RegEx> re;
    re.instantiate();

    // 擦块注释 /* ... */
    re->compile("/\\*[\\s\\S]*?\\*/");
    text = re->sub(text, "", true);

    // 擦行注释 // ...
    re->compile("//.*");
    text = re->sub(text, "", true);

    // 扫 local_size（工作组大小）
    //  匹配格式：layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    //  这三个值决定"一个工作组有多少线程"
    //  配合 image 尺寸 → 算出"派多少工作组"
    re->compile("local_size_x\\s*=\\s*(\\d+)\\s*,\\s*local_size_y\\s*=\\s*(\\d+)\\s*,\\s*local_size_z\\s*=\\s*(\\d+)");
    Ref<RegExMatch> m = re->search(text);
    if (m.is_valid())
    {
        pd.local_size.x = m->get_string(1).to_int();
        pd.local_size.y = m->get_string(2).to_int();
        pd.local_size.z = m->get_string(3).to_int();
        LOG_I("[GLSL解析] local_size = (", pd.local_size.x, ", ", pd.local_size.y, ", ", pd.local_size.z, ")");
    }
    else
    {
        LOG_W("[警告][GLSL解析] 未找到 local_size，用默认 (1,1,1)");
    }

#pragma region 正则image2D
    // 扫 image2D
    //  第一步：抓整体结构 layout(...) [modifiers] image2D name;
    //  第二步：拆解 layout 里的 set / binding / format
    //  第三步：拆解 modifiers 里的 readonly / writeonly

    re->compile("layout\\s*\\(([^)]*)\\)([^;]*?)image2D\\s+([\\w]+)\\s*;");
    m = re->search(text);
    while (m.is_valid())
    {
        String layout_body = m->get_string(1); // "set = 0, binding = 0, rgba32f"
        String modifiers = m->get_string(2);   // "uniform readonly "
        String name = m->get_string(3);        // "tex"

        Dictionary info;
        info["type"] = "image2D";
        info["set"] = 0;
        info["binding"] = 0;
        info["format"] = "";
        info["access"] = "";

        // ---- 拆 set ----
        Ref<RegEx> re_set;
        re_set.instantiate();
        re_set->compile("set\\s*=\\s*(\\d+)");
        Ref<RegExMatch> m_set = re_set->search(layout_body);
        if (m_set.is_valid())
        {
            info["set"] = m_set->get_string(1).to_int();
        }

        // ---- 拆 binding ----
        Ref<RegEx> re_bind;
        re_bind.instantiate();
        re_bind->compile("binding\\s*=\\s*(\\d+)");
        Ref<RegExMatch> m_bind = re_bind->search(layout_body);
        if (m_bind.is_valid())
        {
            info["binding"] = m_bind->get_string(1).to_int();
        }

        // ---- 拆 format ----
        // 格式是 layout 里"不是 set=... 也不是 binding=..."的那个标识符
        // 常见格式：rgba32f / rg32f / r32f / rgba16f / rgba8 等
        Ref<RegEx> re_fmt;
        re_fmt.instantiate();
        re_fmt->compile("([a-z]+\\d+[a-z]*)");
        Ref<RegExMatch> m_fmt = re_fmt->search(layout_body);
        while (m_fmt.is_valid())
        {
            String token = m_fmt->get_string(1);
            // 跳过 set / binding 里的单词
            if (token != "set" && token != "binding")
            {
                info["format"] = token;
                break;
            }
            m_fmt = re_fmt->search(layout_body, m_fmt->get_end());
        }

        // ---- 拆 access ----
        if (modifiers.find("readonly") != -1)
        {
            info["access"] = "readonly";
        }
        else if (modifiers.find("writeonly") != -1)
        {
            info["access"] = "writeonly";
        }

        pd.binding_map[name] = info;
        LOG("[GLSL解析] image2D: ", name,
            " set=", info["set"],
            " binding=", info["binding"],
            " format=", info["format"],
            " access=", info["access"]);

        m = re->search(text, m->get_end());
    }
#pragma endregion
#pragma region 存储buffer
    // 扫 buffer
    // 两步法：先抓整体，再拆 layout 和 modifiers
    re->compile("layout\\s*\\(([^)]*)\\)([^;]*?)buffer\\s+\\w+\\s*\\{[^}]*\\}\\s*([\\w]+)\\s*;");
    m = re->search(text);
    while (m.is_valid())
    {
        String layout_body = m->get_string(1); // "set = 0, binding = 6, std430"
        String modifiers = m->get_string(2);   // "buffer " 或 "readonly buffer "
        String name = m->get_string(3);        // "particles"

        Dictionary info;
        info["type"] = "buffer";
        info["set"] = 0;
        info["binding"] = 0;
        info["format"] = "";
        info["access"] = "";

        // 拆 set
        Ref<RegEx> re_set;
        re_set.instantiate();
        re_set->compile("set\\s*=\\s*(\\d+)");
        Ref<RegExMatch> m_set = re_set->search(layout_body);
        if (m_set.is_valid())
        {
            info["set"] = m_set->get_string(1).to_int();
        }

        // 拆 binding
        Ref<RegEx> re_bind;
        re_bind.instantiate();
        re_bind->compile("binding\\s*=\\s*(\\d+)");
        Ref<RegExMatch> m_bind = re_bind->search(layout_body);
        if (m_bind.is_valid())
        {
            info["binding"] = m_bind->get_string(1).to_int();
        }

        // 拆 format（std430 / std140 / std_scalar 等）
        Ref<RegEx> re_fmt;
        re_fmt.instantiate();
        re_fmt->compile("(std\\w+)");
        Ref<RegExMatch> m_fmt = re_fmt->search(layout_body);
        if (m_fmt.is_valid())
        {
            info["format"] = m_fmt->get_string(1);
        }

        // 拆 access
        if (modifiers.find("readonly") != -1)
        {
            info["access"] = "readonly";
        }
        else if (modifiers.find("writeonly") != -1)
        {
            info["access"] = "writeonly";
        }

        pd.binding_map[name] = info;
        LOG("[GLSL解析] buffer: ", name,
            " set=", info["set"],
            " binding=", info["binding"],
            " format=", info["format"],
            " access=", info["access"]);

        m = re->search(text, m->get_end());
    }
#pragma endregion

#pragma region 只读buffer

    // 扫 uniform_buffer
    // 匹配格式：layout(set = 0, binding = 0, std140) uniform Name { ... } instance;
    // 注意：要和 push_constant 区分——layout 里有 "push_constant" 的跳过
    re->compile("layout\\s*\\(([^)]*)\\)([^;]*?)uniform\\s+\\w+\\s*\\{([^}]*)\\}\\s*([\\w]+)\\s*;");
    m = re->search(text);
    while (m.is_valid())
    {
        String layout_body = m->get_string(1); // "set = 0, binding = 0, std140"
        String modifiers = m->get_string(2);   // "uniform " 或 "readonly uniform "
        String body = m->get_string(3);        // 花括号里的字段
        String name = m->get_string(4);        // 实例名

        // 跳过 push_constant
        if (layout_body.find("push_constant") != -1)
        {
            m = re->search(text, m->get_end());
            continue;
        }

        Dictionary info;
        info["type"] = "uniform_buffer";
        info["set"] = 0;
        info["binding"] = 0;
        info["format"] = "";
        info["access"] = ""; // uniform buffer 通常无修饰符

        // 拆 set
        Ref<RegEx> re_set;
        re_set.instantiate();
        re_set->compile("set\\s*=\\s*(\\d+)");
        Ref<RegExMatch> m_set = re_set->search(layout_body);
        if (m_set.is_valid())
            info["set"] = m_set->get_string(1).to_int();

        // 拆 binding
        Ref<RegEx> re_bind;
        re_bind.instantiate();
        re_bind->compile("binding\\s*=\\s*(\\d+)");
        Ref<RegExMatch> m_bind = re_bind->search(layout_body);
        if (m_bind.is_valid())
            info["binding"] = m_bind->get_string(1).to_int();

        // 拆内存布局（std140 / std430）
        Ref<RegEx> re_fmt;
        re_fmt.instantiate();
        re_fmt->compile("(std\\w+)");
        Ref<RegExMatch> m_fmt = re_fmt->search(layout_body);
        if (m_fmt.is_valid())
            info["format"] = m_fmt->get_string(1);

        //  解析字段列表
        Array fields;
        Ref<RegEx> re_field;
        re_field.instantiate();
        re_field->compile("(\\w+)\\s+(\\w+)\\s*;");
        Ref<RegExMatch> mf = re_field->search(body);
        while (mf.is_valid())
        {
            Dictionary f;
            f["type"] = mf->get_string(1);
            f["name"] = mf->get_string(2);
            fields.append(f);
            LOG("[GLSL解析]   uniform_buffer 字段: ", f["type"], " ", f["name"]);
            mf = re_field->search(body, mf->get_end());
        }

        //  算每个字段的 offset 和 size
        int offset = 0;
        int max_align = 4;
        for (int i = 0; i < fields.size(); i++)
        {
            Dictionary f = fields[i];
            String ftype = f["type"];

            int falign = _glsl_align_std140(ftype);
            int fsize = _glsl_size_std140(ftype);

            offset = _align_up(offset, falign);
            f["offset"] = offset;
            f["size"] = fsize;
            fields[i] = f;

            offset += fsize;
            if (falign > max_align)
                max_align = falign;
        }
        int total_size = _align_up(offset, max_align);
        // std140 uniform block 最小 16 字节，且必须是 16 的倍数
        if (total_size < 16)
            total_size = 16;
        else
            total_size = _align_up(total_size, 16);
        info["size"] = total_size;

        LOG("[GLSL解析]   uniform_buffer 总大小 = ", total_size, " 字节");
        for (int i = 0; i < fields.size(); i++)
        {
            Dictionary f = fields[i];
            LOG("[GLSL解析]     ", f["type"], " ", f["name"],
                " offset=", f["offset"], " size=", f["size"]);
        }

        info["fields"] = fields;

        pd.binding_map[name] = info;
        LOG("[GLSL解析] uniform_buffer: ", name,
            " set=", info["set"],
            " binding=", info["binding"],
            " format=", info["format"]);

        m = re->search(text, m->get_end());
    }

#pragma endregion

#pragma region 正则push
    //  扫 push_constant
    //  匹配格式：layout(push_constant[, std430]) uniform NAME { 字段... } instance;
    //  抓两部分：括号里的字段区 + 实例名
    re->compile("layout\\s*\\(\\s*push_constant\\s*(?:,\\s*([\\w]+)\\s*)?\\)\\s*uniform\\s+[\\w]+\\s*\\{([^}]*)\\}\\s*([\\w]+)\\s*;");

    m = re->search(text);
    if (m.is_valid())
    {
        pd.has_push_const = true;
        String body = m->get_string(2); // {...} 里的内容

        // 解析字段列表：每个字段是 "类型 名字;"
        pd.push_const_decls.clear();
        Ref<RegEx> re_field;
        re_field.instantiate();
        re_field->compile("(\\w+)\\s+(\\w+)\\s*;");
        Ref<RegExMatch> mf = re_field->search(body);
        while (mf.is_valid())
        {
            Dictionary f;
            f["type"] = mf->get_string(1);
            f["name"] = mf->get_string(2);
            pd.push_const_decls.append(f);
            LOG("[GLSL解析] push_constant 字段: ", f["type"], " ", f["name"]);
            mf = re_field->search(body, mf->get_end());
        }
        // 算每个字段的 offset 和 size，以及整个 push_constant 的总大小
        int offset = 0;
        int max_align = 4;
        for (int i = 0; i < pd.push_const_decls.size(); i++)
        {
            Dictionary d = pd.push_const_decls[i];
            String ftype = d["type"];

            int falign = _glsl_align_std430(ftype);
            int fsize = _glsl_size_std430(ftype);

            offset = _align_up(offset, falign); // 对齐到本字段对齐
            d["offset"] = offset;
            d["size"] = fsize;
            pd.push_const_decls[i] = d;

            offset += fsize;
            if (falign > max_align)
                max_align = falign;
        }
        pd.push_const_size = _align_up(offset, max_align); // 整体对齐
        LOG("[GLSL解析] push_constant 总大小 = ", pd.push_const_size, " 字节");
        for (int i = 0; i < pd.push_const_decls.size(); i++)
        {
            Dictionary d = pd.push_const_decls[i];
            LOG("[GLSL解析]   ", d["type"], " ", d["name"],
                " offset=", d["offset"], " size=", d["size"]);
        }
    }
#pragma endregion
#pragma region 正则 特化常量
    // 扫特化常量
    // 匹配格式：layout(constant_id = N) const TYPE NAME = VALUE;
    re->compile("layout\\s*\\(\\s*constant_id\\s*=\\s*(\\d+)\\s*\\)\\s*const\\s+(\\w+)\\s+(\\w+)\\s*=\\s*([^;]+);");

    m = re->search(text);
    pd.spec_const_decls.clear();
    while (m.is_valid())
    {
        Dictionary d;
        d["id"] = m->get_string(1).to_int();
        d["type"] = m->get_string(2);
        d["name"] = m->get_string(3);
        d["default"] = m->get_string(4).strip_edges();

        pd.spec_const_decls.append(d);
        LOG("[GLSL解析] spec_const id=", d["id"],
            " type=", d["type"],
            " name=", d["name"],
            " default=", d["default"]);

        m = re->search(text, m->get_end());
    }
#pragma endregion

    LOG("[GLSL解析] 完成，共 ", pd.binding_map.size(), " 个绑定");
}
#pragma endregion

// 在所有 pass 里找 name 的绑定，返回 pass 索引
//   找不到返回 -1
int GPUPipeline::_find_pass_with_binding(const String &name) const
{
    for (int i = 0; i < (int)_passes.size(); i++)
    {
        if (_passes[i].binding_map.has(name))
            return i;
    }
    return -1;
}

// link 查找：找出这个名字在 _main_texture_ids 里真正对应的 key
//   自己绑了 → 返回自己
//   自己被 link 到某个名字，那个名字绑了 → 返回那个名字
//   都没绑 → 返回空串
String GPUPipeline::_find_bound_name(const String &name, const Dictionary &table) const
{
    if (table.has(name))
        return name;
    if (_link_map.has(name))
    {
        String other = _link_map[name];
        if (table.has(other))
            return other;
    }
    return String();
}

// ============================================================
// 解析"最新那张"的真实绑定名
//   流程：link 别名 → 查 pingpong → 按 flip 计数选 a/b
//   命中 pingpong 返回 a/b 中最新那张；否则返回 bound_name 本身
// ============================================================
String GPUPipeline::_resolve_latest_name(const String &name, bool *is_pingpong_out) const
{
    if (is_pingpong_out)
        *is_pingpong_out = false;

    String bound_name = _find_bound_name(name, _main_texture_ids);
    if (bound_name.is_empty())
        return String();

    for (int i = 0; i < _pingpong_a_names.size(); i++)
    {
        String an = _pingpong_a_names[i];
        String bn = _pingpong_b_names[i];
        if (an == bound_name || bn == bound_name)
        {
            int cnt = _pair_flip_counts.has(an) ? (int)_pair_flip_counts[an] : 0;
            if (is_pingpong_out)
                *is_pingpong_out = true;
            return (cnt % 2 == 1) ? bn : an;
        }
    }
    return bound_name;
}

void GPUPipeline::link(int pass_a, const String &name_a,
                       int pass_b, const String &name_b)
{
    // ---- 校验 ----
    if (pass_a < 0 || pass_a >= (int)_passes.size())
    {
        LOG_W("[link] pass_a=", pass_a, " 越界（共 ", _passes.size(), " 个 pass）");
        return;
    }
    if (pass_b < 0 || pass_b >= (int)_passes.size())
    {
        LOG_W("[link] pass_b=", pass_b, " 越界（共 ", _passes.size(), " 个 pass）");
        return;
    }
    if (!_passes[pass_a].binding_map.has(name_a))
    {
        LOG_W("[link] '", name_a, "' 未在 pass ", pass_a, " 中声明");
        return;
    }
    if (!_passes[pass_b].binding_map.has(name_b))
    {
        LOG_W("[link] '", name_b, "' 未在 pass ", pass_b, " 中声明");
        return;
    }
    Dictionary info_a = _passes[pass_a].binding_map[name_a];
    Dictionary info_b = _passes[pass_b].binding_map[name_b];
    if ((String)info_a["type"] != (String)info_b["type"])
    {
        LOG_W("[link] 两端类型不同：'", name_a, "' 是 ", info_a["type"],
              "，'", name_b, "' 是 ", info_b["type"]);
        return;
    }
    String t = info_a["type"];
    if (t != "image2D" && t != "buffer" && t != "uniform_buffer")
    {
        LOG_W("[link] 该类型不支持 link：", t);
        return;
    }

    // ---- 已有映射冲突 ----
    if (_link_map.has(name_a) && (String)_link_map[name_a] != name_b)
    {
        LOG_W("[link] '", name_a, "' 已链接到 '",
              (String)_link_map[name_a], "'，跳过");
        return;
    }
    if (_link_map.has(name_b) && (String)_link_map[name_b] != name_a)
    {
        LOG_W("[link] '", name_b, "' 已链接到 '",
              (String)_link_map[name_b], "'，跳过");
        return;
    }

    // ---- 建立双向映射 ----
    _link_map[name_a] = name_b;
    _link_map[name_b] = name_a;

    // ---- 两边都绑了 → 警告（submit 时不会合并）----
    const Dictionary *table = nullptr;
    if (t == "image2D")
        table = &_main_texture_ids;
    else if (t == "buffer")
        table = &_storage_buffers;
    else if (t == "uniform_buffer")
        table = &_uniform_buffers;

    if (table && table->has(name_a) && table->has(name_b))
    {
        LOG_W("[link] '", name_a, "' 和 '", name_b,
              "' 都已绑定不同数据，link 在 submit 时不会生效");
    }
}

// 校验：每个 pass 声明的每个 binding，名字是否在本 pipeline 里有绑定
//   只警告，不阻止投递——用户可能在做"只跑部分 pass"的增量测试
void GPUPipeline::_validate_bindings() const
{
    for (int pi = 0; pi < (int)_passes.size(); pi++)
    {
        const PassData &pd = _passes[pi];
        Array names = pd.binding_map.keys();
        for (int i = 0; i < names.size(); i++)
        {
            String name = names[i];
            Dictionary info = pd.binding_map[name];
            String type = info["type"];

            bool missing = false;
            if (type == "image2D")
            {
                missing = _find_bound_name(name, _main_texture_ids).is_empty();
            }
            else if (type == "buffer")
            {
                missing = _find_bound_name(name, _storage_buffers).is_empty();
            }
            else if (type == "uniform_buffer")
            {
                missing = _find_bound_name(name, _uniform_buffers).is_empty();
            }
            else
            {
                continue; // 未知类型，跳过
            }

            if (missing)
            {
                LOG_W("[绑定校验] pass ", pi, " (", pd.shader_path, ") 的 '",
                      name, "' (", type, ") 未绑定");
            }
        }
    }
}
#pragma region 生成GDS模板

// generate_bindings_code —— 根据解析出的 _binding_map，
// 生成一段 GDScript 绑定模板代码
String GPUPipeline::generate_bindings_code() const
{
    const PassData &pd = _passes[0];
    const Dictionary &binding_map = pd.binding_map;
    const Array &spec_const_decls = pd.spec_const_decls;
    const Array &push_const_decls = pd.push_const_decls;
    bool has_push_const = pd.has_push_const;
    String code;
    code += String::utf8("# ==================== GPGPU 绑定模板 ====================\n");
    code += String::utf8("# 由管线自动生成，按需填写数据\n\n");

    // ---- 收集 image2D 和 buffer ----
    Array img_names;
    Array buf_names;

    Array names = binding_map.keys();
    for (int i = 0; i < names.size(); i++)
    {
        String name = names[i];
        Dictionary info = binding_map[name];
        String type = info["type"];
        if (type == "image2D")
            img_names.append(name);
        else if (type == "buffer")
            buf_names.append(name);
    }

    // ---- 第一遍：算最长声明宽度，供注释对齐 ----
    int max_decl_len = 0;
    for (int i = 0; i < img_names.size(); i++)
    {
        String d = "var " + (String)img_names[i] + ": Image";
        if (d.length() > max_decl_len)
            max_decl_len = d.length();
    }
    for (int i = 0; i < buf_names.size(); i++)
    {
        String d = "var " + (String)buf_names[i] + ": PackedFloat32Array";
        if (d.length() > max_decl_len)
            max_decl_len = d.length();
    }

#pragma region 变量字段

    // ---- spec_const ----
    if (spec_const_decls.size() > 0)
    {
        code += String::utf8("\n# --- 特化常量 spec_const ---\n");
        // 第一遍：算最长 call 长度
        int max_call_len = 0;
        for (int i = 0; i < spec_const_decls.size(); i++)
        {
            Dictionary d = spec_const_decls[i];
            String call = "pipe.set_spec_const(" + String::num_int64(d["id"]) + ", " + (String)d["default"] + ")";
            if (call.length() > max_call_len)
                max_call_len = call.length();
        }
        // 第二遍：生成
        for (int i = 0; i < spec_const_decls.size(); i++)
        {
            Dictionary d = spec_const_decls[i];
            String call = "pipe.set_spec_const(" + String::num_int64(d["id"]) + ", " + (String)d["default"] + ")";
            String cmt = " # " + (String)d["type"] + " " + (String)d["name"];

            String line = call;
            int pad = max_call_len + 2 - call.length();
            if (pad < 1)
                pad = 1;
            for (int j = 0; j < pad; j++)
                line += " ";
            code += line + cmt + "\n";
        }
    }

    // ---- image2D ----
    if (img_names.size() > 0)
    {
        code += String::utf8("# --- 纹理 image2D ---\n");
        for (int i = 0; i < img_names.size(); i++)
        {
            String name = img_names[i];
            Dictionary info = binding_map[name];
            String access = info["access"];
            if (access == "")
                access = "read_write";

            String decl = "var " + name + ": Image";
            String cmt = " # set=" + String::num_int64(info["set"]) + " binding=" + String::num_int64(info["binding"]) + " " + (String)info["format"] + " " + access;

            // 补齐到 max_decl_len + 1 列后再放注释
            String line = decl;
            int pad = max_decl_len + 2 - decl.length();
            if (pad < 1)
                pad = 1;
            for (int j = 0; j < pad; j++)
                line += " ";
            code += line + cmt + "\n";
        }
        code += "\n";
    }

    // ---- storage buffer ----
    if (buf_names.size() > 0)
    {
        code += String::utf8("# --- 存储缓冲 storage buffer ---\n");
        for (int i = 0; i < buf_names.size(); i++)
        {
            String name = buf_names[i];
            Dictionary info = binding_map[name];
            String access = info["access"];
            if (access == "")
                access = "read_write";

            String decl = "var " + name + ": PackedFloat32Array";
            String cmt = " # set=" + String::num_int64(info["set"]) + " binding=" + String::num_int64(info["binding"]) + " " + (String)info["format"] + " " + access;

            String line = decl;
            int pad = max_decl_len + 2 - decl.length();
            if (pad < 1)
                pad = 1;
            for (int j = 0; j < pad; j++)
                line += " ";
            code += line + cmt + "\n";
        }
        code += "\n";
    }

    // ---- push_constant 字段（每个字段一个独立变量） ----
    if (has_push_const && push_const_decls.size() > 0)
    {
        code += String::utf8("# --- push_constant 字段 ---\n");
        for (int i = 0; i < push_const_decls.size(); i++)
        {
            Dictionary d = push_const_decls[i];
            String ftype = d["type"];
            String fname = d["name"];

            // 按类型给默认值
            String gd_type;
            String default_val;
            if (ftype == "int" || ftype == "uint")
            {
                gd_type = "int";
                default_val = "0";
            }
            else if (ftype == "float")
            {
                gd_type = "float";
                default_val = "0.0";
            }
            else if (ftype == "bool")
            {
                gd_type = "bool";
                default_val = "false";
            }
            else if (ftype == "vec2")
            {
                gd_type = "Vector2";
                default_val = "Vector2(0, 0)";
            }
            else if (ftype == "vec3")
            {
                gd_type = "Vector3";
                default_val = "Vector3(0, 0, 0)";
            }
            else if (ftype == "vec4")
            {
                gd_type = "Vector4";
                default_val = "Vector4(0, 0, 0, 0)";
            }
            else if (ftype == "ivec2")
            {
                gd_type = "Vector2i";
                default_val = "Vector2i(0, 0)";
            }
            else if (ftype == "ivec3")
            {
                gd_type = "Vector3i";
                default_val = "Vector3i(0, 0, 0)";
            }
            else if (ftype == "ivec4")
            {
                gd_type = "Vector4i";
                default_val = "Vector4i(0, 0, 0, 0)";
            }
            else
            {
                // 未知类型，跳过
                continue;
            }

            // 变量名前缀 pc_，避免和其他绑定重名
            code += "var pc_" + fname + ": " + gd_type + " = " + default_val + "\n";
        }
        code += "\n";
    }

    // ---- uniform buffer 字段 ----
    // 收集所有 uniform_buffer
    Array ubuf_names;
    for (int i = 0; i < names.size(); i++)
    {
        String n = names[i];
        Dictionary inf = binding_map[n];
        if ((String)inf["type"] == "uniform_buffer")
        {
            ubuf_names.append(n);
        }
    }

    if (ubuf_names.size() > 0)
    {
        code += String::utf8("# --- uniform buffer 字段 ---\n");
        for (int i = 0; i < ubuf_names.size(); i++)
        {
            String bname = ubuf_names[i];
            Dictionary inf = binding_map[bname];
            Array fields = inf.get("fields", Array());

            code += "# " + bname + ": set=" + String::num_int64(inf["set"]) + " binding=" + String::num_int64(inf["binding"]) + " " + (String)inf["format"] + "\n";

            for (int j = 0; j < fields.size(); j++)
            {
                Dictionary f = fields[j];
                String ftype = f["type"];
                String fname = f["name"];
                String var_name = "ub_" + bname + "_" + fname;
                String gd_type;
                String default_val;

                if (ftype == "int" || ftype == "uint")
                {
                    gd_type = "int";
                    default_val = "0";
                }
                else if (ftype == "float")
                {
                    gd_type = "float";
                    default_val = "0.0";
                }
                else if (ftype == "bool")
                {
                    gd_type = "bool";
                    default_val = "false";
                }
                else if (ftype == "vec2")
                {
                    gd_type = "Vector2";
                    default_val = "Vector2(0, 0)";
                }
                else if (ftype == "vec3")
                {
                    gd_type = "Vector3";
                    default_val = "Vector3(0, 0, 0)";
                }
                else if (ftype == "vec4")
                {
                    gd_type = "Vector4";
                    default_val = "Vector4(0, 0, 0, 0)";
                }
                else if (ftype == "mat4")
                {
                    gd_type = "PackedFloat32Array";
                    default_val = "PackedFloat32Array()"; // 用户自己填 16 个
                }
                else
                {
                    continue;
                }

                code += "var " + var_name + ": " + gd_type + " = " + default_val + "\n";
            }
            code += "\n";
        }
    }
#pragma endregion
#pragma region 绑定调用
    // ---- 绑定调用 ----
    code += String::utf8("# --- 绑定调用 ---\n");
    for (int i = 0; i < img_names.size(); i++)
    {
        String name = img_names[i];
        code += "pipe.set_image(\"" + name + "\", " + name + ")\n";
    }
    for (int i = 0; i < buf_names.size(); i++)
    {
        String name = buf_names[i];
        code += "pipe.set_storage_buffer(\"" + name + "\", " + name + ")\n";
    }
    // push_constant 调用
    if (has_push_const && push_const_decls.size() > 0)
    {
        String args = "[";
        for (int i = 0; i < push_const_decls.size(); i++)
        {
            Dictionary d = push_const_decls[i];
            if (i > 0)
                args += ", ";
            args += "pc_" + (String)d["name"];
        }
        args += "]";
        code += "pipe.set_push_const(" + args + ")\n";
    }

    // uniform buffer 调用
    for (int i = 0; i < ubuf_names.size(); i++)
    {
        String bname = ubuf_names[i];
        Dictionary inf = binding_map[bname];
        Array fields = inf.get("fields", Array());

        String args = "[";
        for (int j = 0; j < fields.size(); j++)
        {
            Dictionary f = fields[j];
            String var_name = "ub_" + bname + "_" + (String)f["name"];
            if (j > 0)
                args += ", ";
            args += var_name;
        }
        args += "]";

        code += "pipe.set_uniform_buffer(\"" + bname + "\", " + args + ")\n";
    }
#pragma endregion
    code += String::utf8("# ========================================================\n");
    return code;
}

#pragma endregion

#pragma region 绑定
void GPUPipeline::_bind_methods()
{

    ClassDB::bind_method(D_METHOD("submit_task"), &GPUPipeline::submit_task);

    ClassDB::bind_method(D_METHOD("set_iterations", "n"), &GPUPipeline::set_iterations);
    ClassDB::bind_method(D_METHOD("set_dispatch_groups", "g", "pass_idx"), &GPUPipeline::set_dispatch_groups, DEFVAL(-1));

    ClassDB::bind_method(D_METHOD("add_pass", "shader_path"), &GPUPipeline::add_pass);
    ClassDB::bind_method(D_METHOD("get_pass_count"), &GPUPipeline::get_pass_count);
    ClassDB::bind_method(D_METHOD("get_pass_shader_path", "pass_idx"), &GPUPipeline::get_pass_shader_path);
    ClassDB::bind_method(D_METHOD("get_all_shader_paths"), &GPUPipeline::get_all_shader_paths);

    ClassDB::bind_method(D_METHOD("set_spec_const", "id", "value", "pass_idx"), &GPUPipeline::set_spec_const, DEFVAL(0));
    ClassDB::bind_method(D_METHOD("set_push_const", "data", "pass_idx"), &GPUPipeline::set_push_const, DEFVAL(0));

    ClassDB::bind_method(D_METHOD("set_image", "name", "img"), &GPUPipeline::set_image);
    ClassDB::bind_method(D_METHOD("get_image", "name"), &GPUPipeline::get_image);

    ClassDB::bind_method(D_METHOD("get_texture_rid", "name"), &GPUPipeline::get_texture_rid);
    ClassDB::bind_method(D_METHOD("set_texture_rid", "name", "rid"), &GPUPipeline::set_texture_rid);

    ClassDB::bind_method(D_METHOD("set_uniform_buffer", "name", "data"), &GPUPipeline::set_uniform_buffer);
    ClassDB::bind_method(D_METHOD("mark_dirty_uniform_buffer", "name"), &GPUPipeline::mark_dirty_uniform_buffer);

    ClassDB::bind_method(D_METHOD("set_storage_buffer", "name", "data"), &GPUPipeline::set_storage_buffer);
    ClassDB::bind_method(D_METHOD("mark_dirty_storage_buffer", "name"), &GPUPipeline::mark_dirty_storage_buffer);

    ClassDB::bind_method(D_METHOD("set_pingpong", "a_name", "b_name"), &GPUPipeline::set_pingpong);
    ClassDB::bind_method(D_METHOD("get_pingpong_output", "name"), &GPUPipeline::get_pingpong_output);

    ClassDB::bind_method(D_METHOD("generate_bindings_code"), &GPUPipeline::generate_bindings_code);

    ADD_SIGNAL(MethodInfo("task_completed"));
    ADD_SIGNAL(MethodInfo("storage_buffer_ready", PropertyInfo(Variant::STRING, "name"), PropertyInfo(Variant::NIL, "data")));

    ClassDB::bind_method(D_METHOD("request_storage_buffer", "name"), &GPUPipeline::request_storage_buffer);

    ClassDB::bind_method(D_METHOD("link", "pass_a", "name_a", "pass_b", "name_b"), &GPUPipeline::link);
    ClassDB::bind_method(D_METHOD("get_latest_t2drd", "name"), &GPUPipeline::get_latest_t2drd);

    ClassDB::bind_method(D_METHOD("create_display", "source_name", "width", "height"), &GPUPipeline::create_display);
    ClassDB::bind_method(D_METHOD("get_display_texture"), &GPUPipeline::get_display_texture);
}
#pragma endregion

#pragma region 构造函数

GPUPipeline::GPUPipeline() {}
GPUPipeline::~GPUPipeline()
{
    // 逐个释放本管线持有的纹理引用
    // 注册表内部按 ref_count / external 决定是否真正 free_rid
    Array tex_names = _main_texture_ids.keys();
    for (int i = 0; i < tex_names.size(); i++)
    {
        String name = tex_names[i];
        int64_t tex_id = (int64_t)_main_texture_ids[name];
        GPGPU::release_texture(tex_id);
    }
    _main_texture_ids.clear();

    // 清理 display 三缓冲
    for (int i = 0; i < DISPLAY_COUNT; i++)
    {
        if (_display_tex_ids[i] >= 0)
        {
            GPGPU::release_texture(_display_tex_ids[i]);
            _display_tex_ids[i] = -1;
        }
    }
    _display_t2drd = Ref<Texture2DRD>();

    // 通知 worker 自己死了,清理该管线的本地缓存
    // 投递 cleanup 任务给 worker，释放该管线在 worker 侧分配的所有本地 RID
    //   （纹理引用、buffer、pipeline、uniform_set 等）
    Ref<GPUTask> task;
    task.instantiate();
    task->task_type = "cleanup";
    task->pipeline_id = get_instance_id();
    GPGPU::push_task(task);
}

#pragma endregion

#pragma region 计算任务
void GPUPipeline::submit_task()
{
    _validate_bindings(); // 投递前校验
    // 打包完整任务给 worker
    Ref<GPUTask> task;
    task.instantiate();
    task->pipeline_id = get_instance_id(); // 每个 pipeline 对象的唯一 id
    task->iterations = _iterations;        // 计算次数

    // ---- 打包每个 pass ----
    for (int pi = 0; pi < (int)_passes.size(); pi++)
    {
        PassData &pd = _passes[pi];
        Dictionary pt;
        pt["shader_path"] = pd.shader_path;
        pt["local_size"] = pd.local_size;
        pt["push_const"] = pd.push_const_bytes;
        pt["spec_consts"] = pd.spec_constants;
        pt["has_push_const"] = pd.has_push_const;

        // ---- 纹理（遍历本 pass 声明的 image2D）----
        Array pass_tex;
        Array decl_names = pd.binding_map.keys();
        for (int i = 0; i < decl_names.size(); i++)
        {
            String name = decl_names[i];
            Dictionary binding_info = pd.binding_map[name];
            if ((String)binding_info["type"] != "image2D")
                continue;

            // link 查找：别名 → 真名
            String bound_name = _find_bound_name(name, _main_texture_ids);
            if (bound_name.is_empty())
            {
                // 没绑定，_validate_bindings 已经警告过，这里静默跳过
                continue;
            }

            int64_t tex_id = (int64_t)_main_texture_ids[bound_name];
            uint64_t handle = GPGPU::get_texture_handle(tex_id);
            Dictionary tex_info = GPGPU::get_texture_info(tex_id);

            if (handle == 0)
            {
                LOG_W("[警告] submit_task: 纹理 '", name, "' handle 无效，跳过");
                continue;
            }
            int w = (int)tex_info["width"];
            int h = (int)tex_info["height"];
            if (w <= 0 || h <= 0)
            {
                LOG_W("[警告] submit_task: 纹理 '", name, "' 尺寸非法，跳过");
                continue;
            }

            Dictionary ti;
            ti["name"] = bound_name; // ← 用真实绑定的名字，让 worker 合并
            ti["handle"] = handle;
            ti["set"] = binding_info["set"]; // 本 pass 自己的 set/binding
            ti["binding"] = binding_info["binding"];
            ti["format"] = binding_info["format"];
            ti["width"] = w;
            ti["height"] = h;
            ti["access"] = binding_info.get("access", "");

            if (!_init_sent.has(tex_id))
            {
                if (_init_images.has(bound_name))
                {
                    Ref<Image> img = _init_images[bound_name];
                    if (img.is_valid())
                    {
                        ti["init_data"] = img->get_data();
                        _init_sent[tex_id] = true;
                    }
                }
            }
            pass_tex.append(ti);
        }
        pt["textures"] = pass_tex;

        //  打包 storage buffer
        Array pass_sb;
        Array decl_names_sb = pd.binding_map.keys();
        for (int i = 0; i < decl_names_sb.size(); i++)
        {
            String name = decl_names_sb[i];
            Dictionary info = pd.binding_map[name];
            if ((String)info["type"] != "buffer")
                continue;
            String bound_name = _find_bound_name(name, _storage_buffers);
            if (bound_name.is_empty())
                continue;

            Dictionary sb;
            sb["name"] = bound_name;
            sb["set"] = info["set"];
            sb["binding"] = info["binding"];
            sb["access"] = info["access"];

            // 脏时才带 data；不脏只带元信息
            if (_dirty_storage_buffers.has(bound_name))
            {
                Dictionary entry = _storage_buffers[bound_name];
                sb["type"] = entry["type"];
                sb["data"] = entry["data"];
            }

            pass_sb.append(sb);
        }
        pt["storage_buffers"] = pass_sb;

        //  打包 uniform buffer
        Array pass_ub;
        Array decl_names_ub = pd.binding_map.keys();
        for (int i = 0; i < decl_names_ub.size(); i++)
        {
            String name = decl_names_ub[i];
            Dictionary info = pd.binding_map[name];
            if ((String)info["type"] != "uniform_buffer")
                continue;
            String bound_name = _find_bound_name(name, _uniform_buffers);
            if (bound_name.is_empty())
                continue;

            Dictionary ub;
            ub["name"] = bound_name;
            ub["set"] = info["set"];
            ub["binding"] = info["binding"];

            if (_dirty_uniform_buffers.has(bound_name))
            {
                Dictionary entry = _uniform_buffers[bound_name];
                ub["type"] = entry["type"];
                ub["data"] = entry["data"];
            }
            pass_ub.append(ub);
        }
        pt["uniform_buffers"] = pass_ub;

        //  计算 dispatch_groups
        //   用户手动设过 → 用用户值
        //   否则 → 从"可写纹理"的尺寸 + local_size 自动算
        // 手动模式要求三分量都 > 0；否则走自动
        bool manual_dispatch = pd.dispatch_groups.x > 0 && pd.dispatch_groups.y > 0 && pd.dispatch_groups.z > 0;
        if (manual_dispatch)
        {
            pt["dispatch_groups"] = pd.dispatch_groups;
        }
        else
        {
            // 找一张"可写"纹理作为尺寸基准
            //   readonly → 排除（输入纹理，尺寸不代表计算范围）
            //   其他（read_write / writeonly）→ 选它
            // 找一张"可写"纹理作为尺寸基准
            String base_name;
            Array decl_names2 = pd.binding_map.keys();
            for (int i = 0; i < decl_names2.size(); i++)
            {
                String name = decl_names2[i];
                Dictionary info = pd.binding_map[name];
                if ((String)info["type"] != "image2D")
                    continue;
                String access = info.get("access", "");
                if (access != "readonly")
                {
                    base_name = name;
                    break;
                }
            }
            // 全是 readonly 或者没纹理 → 兜底取第一张
            if (base_name.is_empty())
            {
                for (int i = 0; i < decl_names2.size(); i++)
                {
                    String name = decl_names2[i];
                    Dictionary info = pd.binding_map[name];
                    if ((String)info["type"] == "image2D")
                    {
                        base_name = name;
                        break;
                    }
                }
            }

            if (!base_name.is_empty())
            {
                String bound_name = _find_bound_name(base_name, _main_texture_ids);
                if (!bound_name.is_empty())
                {
                    int64_t tid = (int64_t)_main_texture_ids[bound_name];
                    Dictionary tinfo = GPGPU::get_texture_info(tid);
                    int w = (int)tinfo["width"];
                    int h = (int)tinfo["height"];
                    int lsx = pd.local_size.x > 0 ? pd.local_size.x : 1;
                    int lsy = pd.local_size.y > 0 ? pd.local_size.y : 1;
                    pt["dispatch_groups"] = Vector3i(
                        (w + lsx - 1) / lsx,
                        (h + lsy - 1) / lsy,
                        1);
                }
                else
                {
                    pt["dispatch_groups"] = Vector3i(1, 1, 1);
                }
            }
            else
            {
                pt["dispatch_groups"] = Vector3i(1, 1, 1);
            }
        }

        task->passes.append(pt);
    }
    // ---- 所有 pass 打包完，统一清脏 ----
    _dirty_storage_buffers.clear();
    _dirty_uniform_buffers.clear();

    // ---- 打包 ping-pong ----
    //   只打包"两端纹理都已绑定"的对
    //   否则 worker 会按 ping-pong 模式跑，但 set_map 里缺一边 → uniform_set 报错
    for (int i = 0; i < _pingpong_a_names.size(); i++)
    {
        String a_name = _pingpong_a_names[i];
        String b_name = _pingpong_b_names[i];

        if (!_main_texture_ids.has(a_name) || !_main_texture_ids.has(b_name))
        {
            LOG_W("[警告] ping-pong 对 ", a_name, " <-> ", b_name,
                  " 没准备好（两端都要 set_image 过），跳过");
            continue;
        }

        Dictionary p;
        p["a_name"] = a_name;
        p["b_name"] = b_name;
        task->pingpong_pairs.append(p);
    }

    // ---- 投递 ----
    GPGPU::push_task(task);
    Dictionary pt0 = task->passes[0];
    Array texs0 = pt0["textures"];
    LOG_I("[Pipeline] 任务已投递：", _passes[0].shader_path,
          " pass 数=", task->passes.size(),
          " 纹理数=", texs0.size());

    // 标记"可能被写入"的纹理为待读回
    //   readonly → 输入纹理，GPU 侧没变化，不标脏
    //   writeonly / read_write（空）→ 可能被写，标脏
    Dictionary seen_names;
    for (int pi = 0; pi < (int)_passes.size(); pi++)
    {
        Array decl_names = _passes[pi].binding_map.keys();
        for (int i = 0; i < decl_names.size(); i++)
        {
            String name = decl_names[i];
            Dictionary info = _passes[pi].binding_map[name];
            if ((String)info["type"] != "image2D")
                continue;

            String bound_name = _find_bound_name(name, _main_texture_ids);
            if (bound_name.is_empty())
                continue;

            // 同一个 bound_name 只判一次
            if (seen_names.has(bound_name))
                continue;
            seen_names[bound_name] = true;

            // 只要任一 pass 里这个名字（或其别名）可写 → 标脏
            bool is_input_only = true;
            for (int pj = 0; pj < (int)_passes.size(); pj++)
            {
                Array ns = _passes[pj].binding_map.keys();
                for (int k = 0; k < ns.size(); k++)
                {
                    String n = ns[k];
                    if (_find_bound_name(n, _main_texture_ids) != bound_name)
                        continue;
                    Dictionary inf = _passes[pj].binding_map[n];
                    if ((String)inf["type"] != "image2D")
                        continue;
                    String acc = inf.get("access", "");
                    if (acc != "readonly")
                    {
                        is_input_only = false;
                        break;
                    }
                }
                if (!is_input_only)
                    break;
            }
            if (!is_input_only)
                _pending_readback[bound_name] = true;
        }
    }
}

#pragma region 计算变量

// 设置迭代次数
void GPUPipeline::set_iterations(int n)
{
    // 上限保护:防止用户填极端值导致 CPU 内存爆炸
    //   上限 500K 是经验值, 需要更多迭代时,多次 dispatch 即可
    _iterations = CLAMP(n, 1, 500000);
    if (n != _iterations)
    {
        LOG_W("[警告] iterations 被限制到 ", _iterations, "(原值 ", n, ")");
    }
}

// 设置派发工作组数
//    pass_idx = -1 → 全部 pass；否则只设该 pass
//    groups 任一分量 ≤ 0 → 该 pass 恢复"自动"
void GPUPipeline::set_dispatch_groups(const Vector3i &groups, int pass_idx)
{
    // 归一化：任一分量 ≤ 0 → 整组视为"自动"（(0,0,0)）
    Vector3i normalized = groups;
    if (normalized.x <= 0 || normalized.y <= 0 || normalized.z <= 0)
    {
        if (normalized != Vector3i(0, 0, 0))
        {
            LOG_I("dispatch_groups 含非正分量，视为自动模式");
        }
        normalized = Vector3i(0, 0, 0);
    }

    // pass_idx 越界
    if (pass_idx >= (int)_passes.size() || pass_idx < -1)
    {
        LOG_W("[警告] set_dispatch_groups pass_idx=", pass_idx,
              " 越界（pass 数=", _passes.size(), "）");
        return;
    }

    if (pass_idx == -1)
    {
        // 应用到所有 pass
        for (int i = 0; i < (int)_passes.size(); i++)
        {
            _passes[i].dispatch_groups = normalized;
        }
        LOG_I("dispatch_groups 应用到全部 ", _passes.size(), " 个 pass → ", normalized);
    }
    else
    {
        _passes[pass_idx].dispatch_groups = normalized;
        LOG_I("dispatch_groups 应用到 pass ", pass_idx, " → ", normalized);
    }
}

// 加一个 pass
void GPUPipeline::add_pass(const String &shader_path)
{
    PassData p;
    p.shader_path = shader_path;
    _passes.push_back(p);
    LOG_I("[GPUPipeline] 加 pass: ", shader_path, " 共 ", _passes.size(), " 个");

    // 立刻解析这个 pass
    _parse_glsl(_passes.size() - 1);
}

// pass 数量
int GPUPipeline::get_pass_count() const
{
    return (int)_passes.size();
}

// 第 i 个 pass 的 shader 路径（越界返回空串）
String GPUPipeline::get_pass_shader_path(int pass_idx) const
{
    if (pass_idx < 0 || pass_idx >= (int)_passes.size())
        return String();
    return _passes[pass_idx].shader_path;
}

// 全部 pass 的 shader 路径（按 pass 顺序）
PackedStringArray GPUPipeline::get_all_shader_paths() const
{
    PackedStringArray out;
    for (int i = 0; i < (int)_passes.size(); i++)
    {
        out.append(_passes[i].shader_path);
    }
    return out;
}

#pragma endregion

#pragma region 数据绑定

#pragma region 设置push
void GPUPipeline::set_push_const(const Variant &data, int pass_idx)
{
    if (pass_idx < 0 || pass_idx >= (int)_passes.size())
    {
        LOG_W("[警告] set_push_const pass_idx=", pass_idx,
              " 越界（pass 数=", _passes.size(), "）");
        return;
    }
    PassData &pd = _passes[pass_idx];
    // 前提：GLSL 必须有 push_constant 声明 不然你咋GDS设置push
    if (pd.push_const_decls.size() == 0)
    {
        LOG_W("[警告] push_const 未在 GLSL 中声明，忽略");
        return;
    }

    int expected_size = pd.push_const_size;

    // ---- 1. PackedByteArray 手动路径，但仍按 GLSL 检查大小 ----
    if (data.get_type() == Variant::PACKED_BYTE_ARRAY)
    {
        PackedByteArray bytes = data;
        if (bytes.size() != expected_size)
        {
            LOG_W("[警告] push_const 字节数(", bytes.size(),
                  ") 与 GLSL 期望(", expected_size, ") 不匹配");
            return;
        }
        pd.push_const_bytes = bytes;
        LOG_I("push_const 已设（手动字节）：", bytes.size(), " 字节");
        return;
    }

    // ---- 2. 把用户输入转成值列表 ----
    Array values;
    int t = data.get_type();

    if (t == Variant::PACKED_FLOAT32_ARRAY)
    {
        PackedFloat32Array a = data;
        for (int i = 0; i < a.size(); i++)
            values.append(a[i]);
    }
    else if (t == Variant::PACKED_INT32_ARRAY)
    {
        PackedInt32Array a = data;
        for (int i = 0; i < a.size(); i++)
            values.append(a[i]);
    }
    else if (t == Variant::PACKED_FLOAT64_ARRAY)
    {
        PackedFloat64Array a = data;
        for (int i = 0; i < a.size(); i++)
            values.append((float)a[i]);
    }
    else if (t == Variant::PACKED_INT64_ARRAY)
    {
        PackedInt64Array a = data;
        for (int i = 0; i < a.size(); i++)
            values.append((int)a[i]);
    }
    else if (t == Variant::ARRAY)
    {
        values = data;
    }
    else if (t == Variant::VECTOR2 || t == Variant::VECTOR2I)
    {
        Vector2 v = data;
        values.append(v.x);
        values.append(v.y);
    }
    else if (t == Variant::VECTOR3 || t == Variant::VECTOR3I)
    {
        Vector3 v = data;
        values.append(v.x);
        values.append(v.y);
        values.append(v.z);
    }
    else if (t == Variant::VECTOR4 || t == Variant::VECTOR4I)
    {
        Vector4 v = data;
        values.append(v.x);
        values.append(v.y);
        values.append(v.z);
        values.append(v.w);
    }
    else if (t == Variant::INT || t == Variant::FLOAT || t == Variant::BOOL)
    {
        values.append(data);
    }
    else
    {
        LOG_W("[警告] push_const 不支持的类型: ", (int)t);
        return;
    }

    // ---- 3. 数量检查（对 GLSL 字段数） ----
    if (values.size() != pd.push_const_decls.size())
    {
        LOG_W("[警告] push_const 值数量(", values.size(),
              ") 与 GLSL 字段数(", pd.push_const_decls.size(), ") 不匹配");
        return;
    }

    // ---- 4. 按 GLSL 字段类型逐个编码 ----
    PackedByteArray bytes;
    bytes.resize(expected_size);

    for (int i = 0; i < values.size(); i++)
    {
        Dictionary field = pd.push_const_decls[i];
        String ftype = field["type"];
        int foffset = field["offset"];
        Variant v = values[i];

        if (ftype == "int" || ftype == "uint" || ftype == "bool")
        {
            bytes.encode_s32(foffset, (int)v);
        }
        else if (ftype == "float")
        {
            bytes.encode_float(foffset, (float)v);
        }
        else if (ftype == "vec2")
        {
            Vector2 vv = v;
            bytes.encode_float(foffset, vv.x);
            bytes.encode_float(foffset + 4, vv.y);
        }
        else if (ftype == "vec3")
        {
            Vector3 vv = v;
            bytes.encode_float(foffset, vv.x);
            bytes.encode_float(foffset + 4, vv.y);
            bytes.encode_float(foffset + 8, vv.z);
        }
        else if (ftype == "vec4")
        {
            Vector4 vv = v;
            bytes.encode_float(foffset, vv.x);
            bytes.encode_float(foffset + 4, vv.y);
            bytes.encode_float(foffset + 8, vv.z);
            bytes.encode_float(foffset + 12, vv.w);
        }
    }

    pd.push_const_bytes = bytes;
    LOG_I("push_const 已设：", bytes.size(), " 字节");
}

#pragma endregion

#pragma region 设置特化常量
void GPUPipeline::set_spec_const(int id, const Variant &value, int pass_idx)
{
    if (pass_idx < 0 || pass_idx >= (int)_passes.size())
    {
        LOG_W("[警告] set_spec_const pass_idx=", pass_idx,
              " 越界（pass 数=", _passes.size(), "）");
        return;
    }
    PassData &pd = _passes[pass_idx];
    // ---- 1. 找 GLSL 里的声明 ----
    Dictionary decl = _find_spec_decl(pd.spec_const_decls, id);
    if (decl.is_empty())
    {
        LOG_W("[警告] spec_const id=", id, " 未在 GLSL 中声明，忽略");
        return;
    }
    String glsl_type = decl["type"];
    String glsl_name = decl["name"];

    // ---- 2. 检查类型，转成 GLSL 声明的类型 ----
    Dictionary d;
    int t = value.get_type();

    if (glsl_type == "int" || glsl_type == "uint")
    {
        if (t != Variant::INT)
        {
            LOG_W("[警告] spec_const ", glsl_name, " 期望 int/uint，收到类型: ", t);
            return;
        }
        d["type"] = glsl_type;
        d["value"] = (int)value;
    }
    else if (glsl_type == "float")
    {
        if (t != Variant::FLOAT && t != Variant::INT)
        {
            LOG_W("[警告] spec_const ", glsl_name, " 期望 float，收到类型: ", t);
            return;
        }
        d["type"] = "float";
        d["value"] = (float)value;
    }
    else if (glsl_type == "bool")
    {
        if (t != Variant::BOOL)
        {
            LOG_W("[警告] spec_const ", glsl_name, " 期望 bool，收到类型: ", t);
            return;
        }
        d["type"] = "bool";
        d["value"] = (bool)value;
    }
    else
    {
        LOG_W("[警告] spec_const 不支持的类型: ", glsl_type);
        return;
    }

    // ---- 3. 存（内容相同就不覆盖）----
    if (pd.spec_constants.has(id))
    {
        Dictionary old = pd.spec_constants[id];
        if (old["type"] == d["type"] && old["value"] == d["value"])
        {
            LOG_I("spec_const id=", id, " 值未变，跳过");
            return;
        }
    }

    pd.spec_constants[id] = d;
    LOG_I("spec_const id=", id, " (", glsl_name, ") = ", d["value"]);
}

#pragma endregion

#pragma region 设置image
// ============================================================
// 内部：创建 RD 纹理 + 注册到注册表
//   set_image（Image 路径）和 create_display 共用
// ============================================================
int64_t GPUPipeline::_create_rd_texture(int w, int h, const String &fmt,
                                        uint64_t extra_usage, RID *out_rid)
{
    RenderingDevice *main_rd = GPGPU::get_main_rd();
    if (main_rd == nullptr)
    {
        LOG_W("[_create_rd_texture] 主设备未就绪");
        return -1;
    }

    Ref<RDTextureFormat> tf;
    tf.instantiate();
    tf->set_format(_glsl_format_to_rd(fmt));
    tf->set_texture_type(RenderingDevice::TEXTURE_TYPE_2D);
    tf->set_width(w);
    tf->set_height(h);
    tf->set_depth(1);
    tf->set_array_layers(1);
    tf->set_mipmaps(1);
    tf->set_usage_bits(
        RenderingDevice::TEXTURE_USAGE_STORAGE_BIT | RenderingDevice::TEXTURE_USAGE_CAN_UPDATE_BIT | RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT | extra_usage);

    Ref<RDTextureView> tv;
    tv.instantiate();

    // 不传初始数据 —— 内容由 worker compute 决定，避免上传和 worker 写入竞争
    RID rid = main_rd->texture_create(tf, tv);
    if (!rid.is_valid())
    {
        LOG_W("[_create_rd_texture] texture_create 失败");
        return -1;
    }

    uint64_t handle = main_rd->get_driver_resource(
        RenderingDevice::DRIVER_RESOURCE_TEXTURE, rid, 0);

    int64_t tex_id = GPGPU::register_texture(rid, handle, fmt, w, h, false);
    if (tex_id < 0)
    {
        main_rd->free_rid(rid);
        return -1;
    }

    if (out_rid)
        *out_rid = rid;

    LOG_I("[_create_rd_texture] 建纹理 ", w, "x", h, " fmt=", fmt,
          " tex_id=", tex_id, " handle=", handle);
    return tex_id;
}

// set_image —— 接收 Image 或 Texture2D   转换为   Texture2DRD
void GPUPipeline::set_image(const String &name, const Variant &data)
{
    // 从任意 pass 里找 format
    String fmt_str = "rgba32f";
    int pi = _find_pass_with_binding(name);
    if (pi < 0)
    {
        LOG_W("[警告] set_image 名字 '", name, "' 未在任何 pass 中声明");
        return;
    }
    PassData &pd = _passes[pi];
    if (pd.binding_map.has(name))
    {
        Dictionary info = pd.binding_map[name];
        if (info.has("format"))
            fmt_str = info["format"];
    }

    // 1.类型校验
    int type = data.get_type();
    if (type != Variant::OBJECT)
    {
        LOG_W("[警告] set_image 只接收 Image 或 Texture2D");
        return;
    }
    // 判断是 Image 还是 Texture2D
    Ref<Image> img = data;
    Ref<Texture2D> tex = data;
    // 只能是两者一种,都不是跳出
    if (img.is_null() && tex.is_null())
    {
        LOG_W("[警告] set_image 收到不支持的对象类型");
        return;
    }

    // 3.准备新纹理
    RID main_rid;            // 主设备的纹理RID
    uint64_t handle = 0;     // 原生句柄
    int w = 0, h = 0;        // 尺寸
    bool external = true;    // 默认按外部处理，Image 路径会改成 false
    int64_t pre_tex_id = -1; // Image 路径：_create_rd_texture 已注册，记下返回的 tex_id

    if (img.is_valid())
    {
        // ---------- Image 路径：复用 _create_rd_texture ----------
        w = img->get_width();
        h = img->get_height();
        LOG_I("存 Image: ", name, " (", w, "x", h, ")");

        pre_tex_id = _create_rd_texture(w, h, fmt_str, 0, &main_rid);
        if (pre_tex_id < 0)
        {
            LOG_W("[set_image] 创建纹理失败: ", name);
            return;
        }
        handle = GPGPU::get_texture_handle(pre_tex_id);
        external = false; // 自己创建的 → 注册表拥有所有权
    }
    else
    {
        // ---------- Texture2D 路径：借用别人已有的纹理 ----------
        w = tex->get_width();
        h = tex->get_height();
        LOG_I("存 Texture2D: ", name, " (", w, "x", h, ")");

        RID rs_rid = tex->get_rid();

        // 拿主设备 RD 纹理 RID
        main_rid = RenderingServer::get_singleton()->texture_get_rd_texture(rs_rid, false);
        if (!main_rid.is_valid())
        {
            LOG_W("[警告] 无法从 Texture2D 拿 RD 纹理 RID: ", name);
            return;
        }

        // 拿原生句柄
        handle = RenderingServer::get_singleton()->texture_get_native_handle(rs_rid, false);
        external = true; // 别人的 → 注册表只是借用，不拥有
    }

    // 4.注册到登记表
    //    Image 路径：_create_rd_texture 里已注册，直接复用 tex_id
    //    Texture2D 路径：这里才注册（external=true）
    int64_t tex_id;
    if (pre_tex_id >= 0)
    {
        tex_id = pre_tex_id;
    }
    else
    {
        tex_id = GPGPU::register_texture(main_rid, handle, fmt_str, w, h, external);
        if (tex_id < 0)
        {
            LOG_W("[警告] register_texture 失败: ", name);
            return;
        }
    }

    // 5.新纹理就绪,处理旧纹理
    if (_main_texture_ids.has(name))
    {
        int64_t old_tex_id = (int64_t)_main_texture_ids[name];
        GPGPU::release_texture(old_tex_id);
        _main_texture_ids.erase(name);
        _init_sent.erase(old_tex_id); // 清理旧 tex_id 的 init 标记
        _pending_readback.erase(name);
        _init_images.erase(name);     // 清理旧 init 源图
        _readback_images.erase(name); // 清理旧读回缓存
    }

    // 6. 存新状态
    _main_texture_ids[name] = tex_id;

    // Image 路径：存源图供 init_data；未 submit 前 readback 缓存也用源图兜底
    // Texture2D 路径：不存任何 CPU 图（外部纹理无源图，首次 get_image 直接走 GPU 读回）
    if (img.is_valid())
    {
        _init_images[name] = img;
        _readback_images[name] = img;
    }
    else
    {
        _init_images.erase(name);
        _readback_images.erase(name);
    }

    LOG_I("共享纹理就绪: ", name, " ", w, "x", h,
          " handle=", (uint64_t)handle,
          " tex_id=", tex_id,
          " external=", external ? "true" : "false");
}
#pragma endregion
#pragma region 读回image
// 读回显存中的image
Ref<Image> GPUPipeline::get_image(const String &name)
{
    // 遍历所有 pass 找绑定
    int pi = _find_pass_with_binding(name);
    if (pi < 0)
    {
        LOG_W("[警告] get_image 名字 '", name, "' 未在任何 pass 中声明");
        return Ref<Image>();
    }
    PassData &pd = _passes[pi];
    // 名字必须在 GLSL 里
    if (!pd.binding_map.has(name))
    {
        LOG_W("[警告] get_image 名字 '", name, "' 未在 GLSL 中声明");
        return Ref<Image>();
    }

    // ---- link 查找 ----
    String bound_name = _find_bound_name(name, _main_texture_ids);
    if (bound_name.is_empty())
    {
        LOG_W("[警告] get_image '", name, "' 没有绑定纹理");
        return Ref<Image>();
    }

    int64_t tex_id = (int64_t)_main_texture_ids[bound_name];

    RID rid = GPGPU::get_texture_rid(tex_id);
    if (!rid.is_valid())
    {
        LOG_W("[警告] get_image '", bound_name, "' RID 无效");
        return Ref<Image>();
    }
    // 未脏 → 直接用 CPU 缓存，不读回
    if (!_pending_readback.has(bound_name))
    {
        if (_readback_images.has(bound_name))
        {
            Ref<Image> cached = _readback_images[bound_name];
            if (cached.is_valid())
                return cached;
        }
    }

    // 从注册表拿宽高（跨管线传进来的纹理没有 CPU 数据）
    Dictionary info = GPGPU::get_texture_info(tex_id);
    int w = (int)info["width"];
    int h = (int)info["height"];

    if (w <= 0 || h <= 0)
    {
        LOG_W("[警告] get_image '", bound_name, "' 注册表返回尺寸非法: ", w, "x", h);
        return Ref<Image>();
    }

    // ---- 拿格式 ----
    String fmt_str = "rgba32f";
    if (pd.binding_map.has(name))
    {
        Dictionary info = pd.binding_map[name];
        if (info.has("format"))
            fmt_str = info["format"];
    }
    Image::Format img_fmt = _glsl_format_to_image(fmt_str);

    // ---- 从 GPU 读回 ----
    RenderingDevice *rd = GPGPU::get_main_rd();
    if (rd == nullptr)
        return Ref<Image>();

    PackedByteArray bytes = rd->texture_get_data(rid, 0);
    Ref<Image> out_img = Image::create_from_data(w, h, false, img_fmt, bytes);
    if (out_img.is_null())
    {
        LOG_W("[警告] get_image '", name, "' Image::create_from_data 失败，字节数=", bytes.size());
        return Ref<Image>();
    }

    _readback_images[bound_name] = out_img;
    _pending_readback.erase(bound_name);

    LOG_I("读回 '", name, "'：", bytes.size(), " 字节");
    return out_img;
}
#pragma endregion

#pragma region RID传递
RID GPUPipeline::get_texture_rid(const String &name)
{
    String bound_name = _find_bound_name(name, _main_texture_ids);
    if (bound_name.is_empty())
    {
        LOG_W("[警告] get_texture_rid: '", name, "' 未绑定纹理");
        return RID();
    }
    int64_t tex_id = (int64_t)_main_texture_ids[bound_name];
    RID rid = GPGPU::get_texture_rid(tex_id);
    if (!rid.is_valid())
    {
        LOG_W("[警告] get_texture_rid: '", name, "' tex_id=", tex_id, " RID 无效");
        return RID();
    }
    return rid;
}

// 跨管线传 RID：把一个已有的 RID 绑定到本管线的某个绑定名
//   RID 必须已在注册表里（来自另一个 GPUPipeline 或 GPGPU 管理过的纹理）
//   本管线相当于"加一份引用"，不接管所有权
void GPUPipeline::set_texture_rid(const String &name, RID rid)
{
    int pi = _find_pass_with_binding(name);
    if (pi < 0)
    {
        LOG_W("[警告] set_texture_rid 名字 '", name, "' 未在任何 pass 中声明");
        return; // 或 return Ref<Image>() / return false
    }
    PassData &pd = _passes[pi];
    // 1. 名字必须在 GLSL 里声明，且是 image2D
    if (!pd.binding_map.has(name))
    {
        LOG_W("[警告] set_texture_rid: '", name, "' 未在 GLSL 中声明");
        return;
    }
    Dictionary info = pd.binding_map[name];
    if ((String)info["type"] != String("image2D"))
    {
        LOG_W("[警告] set_texture_rid: '", name, "' 类型不是 image2D");
        return;
    }

    // 2. RID 必须有效
    if (!rid.is_valid())
    {
        LOG_W("[警告] set_texture_rid: '", name, "' 收到无效 RID");
        return;
    }

    // 3. 反查注册表，必须已登记
    int64_t tex_id = GPGPU::get_tex_id_by_rid(rid);
    if (tex_id < 0)
    {
        LOG_W("[警告] set_texture_rid: RID=", rid.get_id(),
              " 未在注册表中（不能用裸 RID，必须先经另一个管线管理）");
        return;
    }

    // 4. 如果本管线已经绑定过这个名字，先释放旧的
    if (_main_texture_ids.has(name))
    {
        int64_t old_tex_id = (int64_t)_main_texture_ids[name];
        GPGPU::release_texture(old_tex_id);
        _main_texture_ids.erase(name);
        _init_sent.erase(old_tex_id);
        _pending_readback.erase(name);
    }

    // 5. 加引用，存 tex_id
    GPGPU::add_texture_ref(tex_id);
    _main_texture_ids[name] = tex_id;

    // 6. 清掉 CPU 缓存（这个纹理的 CPU 侧数据不属于本管线）
    _init_images.erase(name);
    _readback_images.erase(name);

    LOG_I("set_texture_rid: ", name, " tex_id=", tex_id,
          " rid=", rid.get_id());
}

// 获取最新的纹理（GPU 侧，零拷贝 Texture2DRD）
//   pingpong 对 → 最新那张
//   非 pingpong → 该纹理本身
Ref<Texture2DRD> GPUPipeline::get_latest_t2drd(const String &name)
{
    String final_name = _resolve_latest_name(name);
    if (final_name.is_empty())
    {
        LOG_W("[警告] get_latest_t2drd: '", name, "' 没有绑定纹理");
        return Ref<Texture2DRD>();
    }
    if (!_main_texture_ids.has(final_name))
    {
        LOG_W("[警告] get_latest_t2drd: '", final_name, "' 未绑定");
        return Ref<Texture2DRD>();
    }
    int64_t tex_id = (int64_t)_main_texture_ids[final_name];
    return GPGPU::get_texture_t2drd(tex_id);
}

// ============================================================
// Display 三缓冲
// ============================================================
void GPUPipeline::create_display(const String &source_name, int width, int height)
{
    if (_display_enabled)
    {
        LOG_W("[create_display] 已启用，忽略重复调用");
        return;
    }
    if (width <= 0 || height <= 0)
    {
        LOG_W("[create_display] 尺寸非法: ", width, "x", height);
        return;
    }

    // 1. 找 source_name 所属的 pingpong 对
    String bound_name = _find_bound_name(source_name, _main_texture_ids);
    if (bound_name.is_empty())
    {
        LOG_W("[create_display] source_name '", source_name, "' 未绑定纹理");
        return;
    }

    String found_a, found_b;
    for (int i = 0; i < _pingpong_a_names.size(); i++)
    {
        String an = _pingpong_a_names[i];
        String bn = _pingpong_b_names[i];
        if (an == bound_name || bn == bound_name)
        {
            found_a = an;
            found_b = bn;
            break;
        }
    }
    if (found_a.is_empty())
    {
        LOG_W("[create_display] source_name '", source_name, "' 不在任何 pingpong 对里");
        return;
    }

    // 2. 从源纹理拿格式
    int64_t src_tex_id = (int64_t)_main_texture_ids[bound_name];
    String fmt_str = (String)GPGPU::get_texture_info(src_tex_id)["format"];

    // 3. 建 3 张 display（复用 _create_rd_texture，加 CAN_COPY_TO_BIT）
    Array display_handles;
    for (int i = 0; i < DISPLAY_COUNT; i++)
    {
        RID rid;
        int64_t tex_id = _create_rd_texture(
            width, height, fmt_str,
            RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT, // worker texture_copy 目标
            &rid);
        if (tex_id < 0)
        {
            LOG_E("[create_display] 创建第 ", i, " 张 display 失败，回滚");
            for (int k = 0; k < i; k++)
            {
                if (_display_tex_ids[k] >= 0)
                {
                    GPGPU::release_texture(_display_tex_ids[k]);
                    _display_tex_ids[k] = -1;
                }
            }
            return;
        }
        _display_tex_ids[i] = tex_id;

        uint64_t handle = GPGPU::get_texture_handle(tex_id);
        display_handles.append((int64_t)handle);
        LOG_I("[create_display] display[", i, "] tex_id=", tex_id);
    }

    // 4. 稳定 Texture2DRD（内部换 RID，对象身份不变）
    _display_t2drd.instantiate();
    _display_t2drd->set_texture_rd_rid(GPGPU::get_texture_rid(_display_tex_ids[0]));

    // 5. 状态
    _display_source_a = found_a;
    _display_source_b = found_b;
    _display_enabled = true;
    _display_current_show = -1;

    // 6. 投递 create_display 任务给 worker
    Ref<GPUTask> task;
    task.instantiate();
    task->pipeline_id = get_instance_id();
    task->task_type = "create_display";
    task->display_handles = display_handles;
    task->display_source_a = found_a;
    task->display_source_b = found_b;
    task->display_width = width;
    task->display_height = height;
    task->display_format = fmt_str;

    GPGPU::push_task(task);
    LOG_I("[create_display] 已投递 create_display 任务，源 ", found_a, " <-> ", found_b);
}

Ref<Texture2DRD> GPUPipeline::get_display_texture()
{
    if (!_display_enabled)
    {
        LOG_W("[get_display_texture] display 未启用，先调 create_display");
        return Ref<Texture2DRD>();
    }
    return _display_t2drd;
}

void GPUPipeline::apply_display_flip(int show_idx)
{
    if (!_display_enabled)
        return;
    if (show_idx < 0 || show_idx >= DISPLAY_COUNT)
    {
        LOG_W("[apply_display_flip] show_idx 非法: ", show_idx);
        return;
    }
    if (show_idx == _display_current_show)
        return;

    _display_current_show = show_idx;
    _display_t2drd->set_texture_rd_rid(GPGPU::get_texture_rid(_display_tex_ids[show_idx]));
    LOG("[apply_display_flip] 切到 display[", show_idx, "]");
}

#pragma endregion

#pragma region 只读 buffer
void GPUPipeline::set_uniform_buffer(const String &name, const Variant &data)
{
    int pi = _find_pass_with_binding(name);
    if (pi < 0)
    {
        LOG_W("[警告] set_uniform_buffer 名字 '", name, "' 未在任何 pass 中声明");
        return; // 或 return Ref<Image>() / return false
    }
    PassData &pd = _passes[pi];
    // 前提：GLSL 里必须有这个 uniform_buffer ----
    if (!pd.binding_map.has(name))
    {
        LOG_W("[警告] set_uniform_buffer 名字 '", name, "' 未在 GLSL 中声明，忽略");
        return;
    }
    Dictionary info = pd.binding_map[name];
    String info_type = info.get("type", "");
    if (info_type != String("uniform_buffer"))
    {
        LOG_W("[警告] set_uniform_buffer 名字 '", name, "' 类型不是 uniform_buffer，忽略");
        return;
    }

    Array fields = info.get("fields", Array());
    int expected_size = info.get("size", 0);

    // ---- 特殊路径：PackedByteArray 手动字节，只检查大小 ----
    if (data.get_type() == Variant::PACKED_BYTE_ARRAY)
    {
        PackedByteArray bytes = data;
        if (bytes.size() != expected_size)
        {
            LOG_W("[警告] set_uniform_buffer 字节数(", bytes.size(),
                  ") 与 GLSL 期望(", expected_size, ") 不匹配");
            return;
        }
        Dictionary entry;
        entry["type"] = "byte";
        entry["data"] = bytes;

        if (_uniform_buffers.has(name))
        {
            Dictionary old = _uniform_buffers[name];
            if (old["type"] == entry["type"] && old["data"] == entry["data"])
            {
                LOG_I("uniform buffer ", name, " 内容未变，跳过标脏");
                return;
            }
        }

        _uniform_buffers[name] = entry;
        _dirty_uniform_buffers[name] = true;
        LOG_I("存 uniform buffer: ", name, " (手动字节 ", bytes.size(), ")");
        return;
    }

    // ---- 主路径：Array，每个元素对应一个字段 ----
    if (data.get_type() != Variant::ARRAY)
    {
        LOG_W("[警告] set_uniform_buffer 只接收 Array 或 PackedByteArray");
        return;
    }
    Array values = data;

    if (values.size() != fields.size())
    {
        LOG_W("[警告] set_uniform_buffer 值数量(", values.size(),
              ") 与 GLSL 字段数(", fields.size(), ") 不匹配");
        return;
    }

    // ---- 按字段类型逐个编码 ----
    PackedByteArray bytes;
    bytes.resize(expected_size);

    for (int i = 0; i < fields.size(); i++)
    {
        Dictionary f = fields[i];
        String ftype = f["type"];
        int foffset = f["offset"];
        Variant v = values[i];

        if (ftype == "int" || ftype == "uint")
        {
            bytes.encode_s32(foffset, (int)v);
        }
        else if (ftype == "float")
        {
            bytes.encode_float(foffset, (float)v);
        }
        else if (ftype == "bool")
        {
            bytes.encode_s32(foffset, (bool)v ? 1 : 0);
        }
        else if (ftype == "vec2")
        {
            Vector2 vv = v;
            bytes.encode_float(foffset, vv.x);
            bytes.encode_float(foffset + 4, vv.y);
        }
        else if (ftype == "vec3")
        {
            Vector3 vv = v;
            bytes.encode_float(foffset, vv.x);
            bytes.encode_float(foffset + 4, vv.y);
            bytes.encode_float(foffset + 8, vv.z);
        }
        else if (ftype == "vec4")
        {
            Vector4 vv = v;
            bytes.encode_float(foffset, vv.x);
            bytes.encode_float(foffset + 4, vv.y);
            bytes.encode_float(foffset + 8, vv.z);
            bytes.encode_float(foffset + 12, vv.w);
        }
        else if (ftype == "mat4")
        {
            // mat4 期望 PackedFloat32Array(16)
            PackedFloat32Array m = v;
            if (m.size() != 16)
            {
                LOG_W("[警告] mat4 需要 16 个 float，实际 ", m.size());
                return;
            }
            for (int j = 0; j < 16; j++)
            {
                bytes.encode_float(foffset + j * 4, m[j]);
            }
        }
        else
        {
            LOG_W("[警告] uniform_buffer 字段类型暂不支持: ", ftype);
            return;
        }
    }

    Dictionary entry;
    entry["type"] = "byte";
    entry["data"] = bytes;

    if (_uniform_buffers.has(name))
    {
        Dictionary old = _uniform_buffers[name];
        if (old["type"] == entry["type"] && old["data"] == entry["data"])
        {
            LOG_I("uniform buffer ", name, " 内容未变，跳过标脏");
            return;
        }
    }

    _uniform_buffers[name] = entry;
    _dirty_uniform_buffers[name] = true;
    LOG_I("存 uniform buffer: ", name, " (按字段编码 ", bytes.size(), " 字节)");
}

#pragma endregion
#pragma region 存储buffer
// set_buffer —— 接收任意支持的数组类型
void GPUPipeline::set_storage_buffer(const String &name, const Variant &data)
{
    int pi = _find_pass_with_binding(name);
    if (pi < 0)
    {
        LOG_W("[警告] set_storage_buffer 名字 '", name, "' 未在任何 pass 中声明");
        return; // 或 return Ref<Image>() / return false
    }
    PassData &pd = _passes[pi];
    // GLSL 名字和类型校验
    int check = _check_binding(pd.binding_map, name, "buffer");
    if (check == 1)
    {
        LOG_W("[警告] set_storage_buffer 名字 '", name, "' 未在 GLSL 中声明，忽略");
        return;
    }
    if (check == 2)
    {
        LOG_W("[警告] set_storage_buffer 名字 '", name, "' 类型不是 buffer，忽略");
        return;
    }

    Dictionary entry;
    int t = data.get_type();

    if (t == Variant::PACKED_FLOAT32_ARRAY)
    {
        entry["type"] = "float32";
        entry["data"] = (PackedFloat32Array)data;
    }
    else if (t == Variant::PACKED_INT32_ARRAY)
    {
        entry["type"] = "int32";
        entry["data"] = (PackedInt32Array)data;
    }
    else if (t == Variant::PACKED_BYTE_ARRAY)
    {
        entry["type"] = "byte";
        entry["data"] = (PackedByteArray)data;
    }
    else
    {
        LOG_W("[警告] set_buffer 不支持的类型: ", (int)t, "（只支持 PackedFloat32Array / PackedInt32Array / PackedByteArray）");
        return;
    }

    // ---- 内容对比：和上次一样就不标脏 ----
    if (_storage_buffers.has(name))
    {
        Dictionary old = _storage_buffers[name];
        if (old["type"] == entry["type"] && old["data"] == entry["data"])
        {
            LOG_I("storage buffer ", name, " 内容未变，跳过标脏");
            return;
        }
    }

    _storage_buffers[name] = entry;
    _dirty_storage_buffers[name] = true;
    LOG("存 storage buffer: ", name, " type=", entry["type"], " 长度=", ((PackedByteArray)entry["data"]).size());
}

// 请求从 GPU 读回 storage buffer
//   投递一个 readback 任务，结果通过 storage_buffer_ready 信号返回
void GPUPipeline::request_storage_buffer(const String &name)
{
    int pi = _find_pass_with_binding(name);
    if (pi < 0)
    {
        LOG_W("[警告] request_storage_buffer 名字 '", name, "' 未在任何 pass 中声明");
        return; // 或 return Ref<Image>() / return false
    }
    PassData &pd = _passes[pi];
    if (!pd.binding_map.has(name))
    {
        LOG_W("[警告] request_storage_buffer '", name, "' 未在 GLSL 中声明");
        return;
    }

    Dictionary info = pd.binding_map[name];
    String info_type = info.get("type", "");
    if (info_type != String("buffer"))
    {
        LOG_W("[警告] request_storage_buffer '", name, "' 类型不是 buffer");
        return;
    }

    Ref<GPUTask> task;
    task.instantiate();
    task->pipeline_id = get_instance_id();
    task->task_type = "readback_buffer";
    task->readback_buffer_name = name;

    GPGPU::push_task(task);
    LOG_I("[Pipeline] 请求读回 buffer: ", name);
}

#pragma endregion
#pragma region ping-pong

// 声明 ping-pong 双缓冲
//   只记录名字——真正的交换逻辑在 dispatch 里
void GPUPipeline::set_pingpong(const String &a_name, const String &b_name)
{
    if (a_name == b_name)
    {
        LOG_W("[警告] set_pingpong a 和 b 不能是同一个名字: ", a_name);
        return;
    }

    int pi_a = _find_pass_with_binding(a_name);
    if (pi_a < 0)
    {
        LOG_W("[警告] set_pingpong 名字 '", a_name, "' 未在任何 pass 中声明");
        return;
    }
    Dictionary info_a = _passes[pi_a].binding_map[a_name];
    String type_a = info_a.get("type", "");
    if (type_a != String("image2D"))
    {
        LOG_W("[警告] set_pingpong 名字 '", a_name, "' 类型不是 image2D");
        return;
    }

    int pi_b = _find_pass_with_binding(b_name);
    if (pi_b < 0)
    {
        LOG_W("[警告] set_pingpong 名字 '", b_name, "' 未在任何 pass 中声明");
        return;
    }
    Dictionary info_b = _passes[pi_b].binding_map[b_name];
    String type_b = info_b.get("type", "");
    if (type_b != String("image2D"))
    {
        LOG_W("[警告] set_pingpong 名字 '", b_name, "' 类型不是 image2D");
        return;
    }

    // ---- 4. 不能重复注册 ----
    for (int i = 0; i < _pingpong_a_names.size(); i++)
    {
        String an = _pingpong_a_names[i];
        String bn = _pingpong_b_names[i];
        if (an == a_name || bn == a_name)
        {
            LOG_W("[警告] set_pingpong '", a_name, "' 已经在 ping-pong 里了");
            return;
        }
        if (an == b_name || bn == b_name)
        {
            LOG_W("[警告] set_pingpong '", b_name, "' 已经在 ping-pong 里了");
            return;
        }
    }

    // ---- 通过，注册 ----
    _pingpong_a_names.append(a_name);
    _pingpong_b_names.append(b_name);
    LOG("ping-pong 已启用: ", a_name, " <-> ", b_name,
        "  共 ", _pingpong_a_names.size(), " 对");
}

// 拿 ping-pong 最终输出（CPU 读回 Image）
//   pingpong 对 → 读回最新那张
//   非 pingpong 名字 → 返回空 Variant（保持原行为，避免误用）
Variant GPUPipeline::get_pingpong_output(const String &name)
{
    bool is_pingpong = false;
    String final_name = _resolve_latest_name(name, &is_pingpong);

    if (final_name.is_empty())
    {
        LOG_W("[警告] get_pingpong_output: '", name, "' 没有绑定纹理");
        return Variant();
    }
    if (!is_pingpong)
    {
        // 非 pingpong，按历史语义返回空（不会静默把单张纹理当 pingpong 结果）
        return Variant();
    }

    _pending_readback[final_name] = true; // 强制 get_image 走 GPU 读回
    Ref<Image> img = get_image(final_name);
    if (img.is_null())
        return Variant();
    return img;
}

void GPUPipeline::set_pair_flip_counts(const Dictionary &counts)
{
    _pair_flip_counts = counts;
}

Dictionary GPUPipeline::get_pair_flip_counts() const
{
    return _pair_flip_counts;
}
#pragma endregion

#pragma region 手动标脏

// 对storage_buffer标脏
void GPUPipeline::mark_dirty_storage_buffer(const String &name)
{
    _dirty_storage_buffers[name] = true;
    LOG_I("mark_dirty_storage_buffer: ", name);
}

// 手动标脏
void GPUPipeline::mark_dirty_uniform_buffer(const String &name)
{
    _dirty_uniform_buffers[name] = true;
    LOG_I("mark_dirty_uniform_buffer: ", name);
}

#pragma endregion

#pragma endregion
