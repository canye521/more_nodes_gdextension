#include "gpgpu.hpp"

#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/engine.hpp>



// 构造函数
GPGPU::GPGPU() {}
GPGPU::~GPGPU() {}


//绑定
void GPGPU::_bind_methods() {
    ClassDB::bind_static_method("GPGPU", D_METHOD("create_pipeline", "shader_paths"),&GPGPU::create_pipeline);
}



RenderingDevice *GPGPU::_main_rd = nullptr;


// 获取主渲染设备对象
RenderingDevice *GPGPU::get_main_rd() {
    if (_main_rd == nullptr) {
        _main_rd = RenderingServer::get_singleton()->get_rendering_device();
        LOG_I("[GPGPU] 获取主渲染设备");
    }
    return _main_rd;
}


//对GDS开放
// 创建一个管线对象，并设置着色器路径
Ref<GPUPipeline> GPGPU::create_pipeline(const Variant &shader_paths) {
    Ref<GPUPipeline> pipe = Ref<GPUPipeline>(memnew(GPUPipeline));

    int path_type = shader_paths.get_type();

    if (path_type == Variant::STRING || path_type == Variant::STRING_NAME) {
        // 单路径
        String path = shader_paths;
        LOG_I("[GPGPU] 创建管线（单 pass）: ", path);
        pipe->add_pass(path);

    } else if (path_type == Variant::ARRAY) {
        Array arr = shader_paths;
        LOG_I("[GPGPU] 创建管线（", arr.size(), " pass）");
        for (int i = 0; i < arr.size(); i++) {
            pipe->add_pass((String)arr[i]);
        }

    } else if (path_type == Variant::PACKED_STRING_ARRAY) {
        PackedStringArray arr = shader_paths;
        LOG_I("[GPGPU] 创建管线（", arr.size(), " pass）");
        for (int i = 0; i < arr.size(); i++) {
            pipe->add_pass(arr[i]);
        }

    } else {
        LOG_W("[GPGPU] create_pipeline 参数类型不支持: ", (int)path_type,
              "（要 String / Array / PackedStringArray）");
    }

    return pipe;
}



#pragma region 线程

Ref<GPGPUWorker> GPGPU::_worker;       //后台线程
bool GPGPU::_poll_connected = false;   //仅用一次,连接轮询

//确保线程存在
Ref<GPGPUWorker> GPGPU::_ensure_worker() {
    if (_worker.is_null()) {
        _worker.instantiate();     
        _worker->start();
        LOG_I("[GPGPU] Worker 线程启动");
    }
    return _worker;
}

//GDE卸载时调用  清空工作线程
void GPGPU::cleanup_worker() {   // 仅GDE反初始化时调用
    if (_worker.is_valid()) {
        _worker = Ref<GPGPUWorker>();
        LOG_I("[GPGPU] Worker 线程关闭");
    }
}


//GPU管线调用   负责确保 worker 就绪、轮询已连接、然后把任务转发给 worker
void GPGPU::push_task(const Ref<GPUTask> &task) {

    //如果从未连接轮询就连接
    if (!_poll_connected) {   
        SceneTree *tree = Object::cast_to<SceneTree>(
            Engine::get_singleton()->get_main_loop());
        if (tree != nullptr) {
            tree->connect("process_frame",
                callable_mp_static(&GPGPU::_on_process_frame));
            _poll_connected = true;
            LOG_I("[GPGPU] 已连接 process_frame 轮询");
        } else {
            LOG_W("[GPGPU] 无法连接 process_frame(main_loop 不是 SceneTree?)");
        }
    }
    _ensure_worker()->push_task(task);     //如果是第一次投任务就创建线程,然后投递任务
}


// 主循环轮询：从 worker 结果队列取标记，emit 到对应 pipeline
void GPGPU::_on_process_frame() {
    if (_worker.is_null()) return;   

    while (_worker->get_result_count() > 0) {  //当线程里有任务时   worker 结果队列的长度
        Dictionary result = _worker->pop_result();  //取队列
        if (result.is_empty()) break;

        int64_t pid = (int64_t)result["pipeline_id"];
        String signal_name = result["signal"];
        Array args = result["args"];

        //从pid反查GPU管线对象
        Object *obj = ObjectDB::get_instance(ObjectID(pid));
        if (obj == nullptr) continue;

        GPUPipeline *pipe = Object::cast_to<GPUPipeline>(obj);
        if (pipe == nullptr) continue;

        //根据任务完成类型输出
        if (signal_name == "task_completed") {
            if (result.has("pair_flip_count")) {
                pipe->set_pair_flip_counts(result["pair_flip_count"]);
            }
            if (result.has("display_show_idx")) {
                int show_idx = (int)result["display_show_idx"];
                if (show_idx >= 0) {
                    pipe->apply_display_flip(show_idx);
                }
            }
            pipe->emit_signal(StringName("task_completed"));
        }
        else if (signal_name == "storage_buffer_ready") {
            // 2 个参数：显式取出来传
            if (args.size() >= 2) {
                Variant v0 = args[0];
                Variant v1 = args[1];
                pipe->emit_signal(StringName("storage_buffer_ready"), v0, v1);
            }
        }
    }
}

#pragma endregion




#pragma region 纹理注册表

/*
 * 注册表容器：延迟到首次调用时构造
 * 写成全局静态（Dictionary _reg_textures;），DLL 加载时就会构造。
 * 但此时 Godot 的 Variant 系统可能还没初始化，Dictionary 构造会崩，
 * 表现为整个 GDExtension 打不开，且没有任何有用报错。
*/
namespace {
    //主表   {tex_id : Dict}
    Dictionary &_reg_textures() {
        static Dictionary d;
        return d;
    }
    //反查表 {  RID  : tex_id}
    Dictionary &_reg_by_rid() {
        static Dictionary d;
        return d;
    }
    // 表页数
    int64_t &_reg_next_id() {
        static int64_t v = 1;
        return v;
    }
}
    #pragma region 初始化
// 初始化注册表：由 register_types.cpp 在 entry_symbol 里调用一次
//   此时 Godot Variant 系统完全就绪，构造 Dictionary 安全
void GPGPU::init_registry() {
    (void)_reg_textures();
    (void)_reg_by_rid();
    (void)_reg_next_id();
}

// 清理：遍历注册表，释放所有自己创建的主设备纹理 RID
//   必须在 cleanup_worker() 之后调用
void GPGPU::cleanup_registry() {
    RenderingDevice *rd = _main_rd;
    if (rd != nullptr) {
        Array keys = _reg_textures().keys();   //取主表所有的键  tex_id
        for (int i = 0; i < keys.size(); i++) {
            int64_t tex_id = keys[i];
            Dictionary entry = _reg_textures()[tex_id];

            bool external = entry["external"];  //false表示是GPGPU所拥有的
            RID rid = entry["rid"];

            // 自己创建的才释放；别人的 RID 无权 free
            if (!external && rid.is_valid()) {
                rd->free_rid(rid);
            }
        }
    }
    //重置注册表
    _reg_textures().clear();
    _reg_by_rid().clear();
    _reg_next_id() = 1;
    LOG_I("[GPGPU] 注册表已清空");
}

    #pragma endregion


    #pragma region 注册引用
// 纹理注册表：登记
//   新建一条 entry，ref_count 初始为 1（登记即代表调用方持有一份）
//   返回 tex_id；外部拿 RID 反查也走这个 id
int64_t GPGPU::register_texture(RID rid,                     //主设备RID
                                uint64_t handle,             //纹理原始句柄
                                const String &format,        //纹理格式  GLSL解析出来的
                                int w, int h,                //宽高
                                bool external) {             //决定是否需要释放RID
    //安全检测
    if (!rid.is_valid()) {
        LOG_W("[GPGPU] register_texture: RID 无效");
        return -1;
    }
    // 重复检查：同一 RID 已经登记过，直接返回已有 tex_id
    int64_t existing = get_tex_id_by_rid(rid);
    if (existing >= 0) {
        add_texture_ref(existing);   
        LOG_I("[GPGPU] register_texture: RID=", rid.get_id(),
              " 已注册为 tex_id=", existing, "，加引用后返回");
        return existing;
    }

    int64_t tex_id = _reg_next_id()++;  //第一个注册的是id = 1

    Dictionary entry;
    entry["rid"]        = rid;
    entry["handle"]     = (int64_t)handle;
    entry["format"]     = format;            //GLSL 格式字符串（"rgba32f" )
    entry["width"]      = w;
    entry["height"]     = h;
    entry["external"]   = external;            //“GPGPU 有没有这个 RID 的所有权
    entry["ref_count"]  = 1;                    //该纹理被引用1次
    
    //创建共享的 Texture2DRD渲染侧包装，所有引用者共用
    Ref<Texture2DRD> t2drd;
    t2drd.instantiate();
    t2drd->set_texture_rd_rid(rid);
    entry["t2drd"] = t2drd;

    _reg_textures()[tex_id] = entry;
    _reg_by_rid()[(int64_t)rid.get_id()] = tex_id;    //写入反查表

    LOG_I("[GPGPU] 注册纹理 tex_id=", tex_id,
          " rid=", rid.get_id(),
          " ", w, "x", h,
          " fmt=", format,
          " external=", external ? "true" : "false");

    return tex_id;
}

// 纹理注册表：加引用
//   另一个 pipeline 声明“我要用这张纹理”
//   ref_count++；不存在则警告
void GPGPU::add_texture_ref(int64_t tex_id) {
    if (!_reg_textures().has(tex_id)) {
        LOG_W("[GPGPU] 添加引用: tex_id=", tex_id, " 不存在");
        return;
    }
    Dictionary entry = _reg_textures()[tex_id];
    int64_t rc = (int64_t)entry["ref_count"];
    rc++;
    entry["ref_count"] = rc;
    _reg_textures()[tex_id] = entry;

    LOG_I("[GPGPU] 添加引用 tex_id=", tex_id, " ref_count=", rc);
}

// 纹理注册表：减引用
//   ref_count--；
//   当 ref_count == 0 且 !external 时，真正 free_rid 并删除记录
//   外部纹理（external=true）永不 free_rid，但仍删记录
void GPGPU::release_texture(int64_t tex_id) {
    if (!_reg_textures().has(tex_id)) {
        LOG_W("[GPGPU] 清除引用: tex_id=", tex_id, " 不存在");
        return;
    }

    Dictionary entry = _reg_textures()[tex_id];
    int64_t rc = (int64_t)entry["ref_count"];
    rc--;

    if (rc > 0) {
        // 还有人在用，只更新计数
        entry["ref_count"] = rc;
        _reg_textures()[tex_id] = entry;
        LOG_I("[GPGPU] 清除引用 tex_id=", tex_id, " ref_count=", rc);
        return;
    }

    //  rc == 0：真正释放 
    bool external = entry["external"];
    RID rid = entry["rid"];

    if (!external && rid.is_valid()) {
        RenderingDevice *rd = get_main_rd();
        if (rd != nullptr) {
            rd->free_rid(rid);
            LOG_I("[GPGPU] 释放主设备纹理 tex_id=", tex_id, " rid=", rid.get_id());
        }
    } else {
        LOG_I("[GPGPU] 外部纹理 tex_id=", tex_id, " 不释放 RID，仅摘除记录");
    }

    // 反查表也要清
    if (rid.is_valid()) {
        _reg_by_rid().erase((int64_t)rid.get_id());
    }

    _reg_textures().erase(tex_id);
}

    #pragma endregion
    #pragma region 查询
// 纹理注册表：反查
//   给一个主设备 RID，找出它在注册表里的 tex_id
//   找不到返回 -1
int64_t GPGPU::get_tex_id_by_rid(RID rid) {
    if (!rid.is_valid()) return -1;
    int64_t key = (int64_t)rid.get_id();
    if (_reg_by_rid().has(key)) return (int64_t)_reg_by_rid()[key];
    return -1;
}

// 纹理注册表：只读查询
//   四个接口都是“给 tex_id，取字段”
//   tex_id 无效时返回各自的空值/默认值
RID GPGPU::get_texture_rid(int64_t tex_id) {
    if (!_reg_textures().has(tex_id)) return RID();
    Dictionary entry = _reg_textures()[tex_id];
    return entry["rid"];
}

uint64_t GPGPU::get_texture_handle(int64_t tex_id) {
    if (!_reg_textures().has(tex_id)) return 0;
    Dictionary entry = _reg_textures()[tex_id];
    return (uint64_t)(int64_t)entry["handle"];
}

Dictionary GPGPU::get_texture_info(int64_t tex_id) {
    if (!_reg_textures().has(tex_id)) return Dictionary();
    Dictionary entry = _reg_textures()[tex_id];

    Dictionary out;
    out["format"]   = entry["format"];
    out["width"]    = entry["width"];
    out["height"]   = entry["height"];
    out["external"] = entry["external"];
    out["ref_count"]= entry["ref_count"];
    return out;
}

Ref<Texture2DRD> GPGPU::get_texture_t2drd(int64_t tex_id) {
    if (!_reg_textures().has(tex_id)) return Ref<Texture2DRD>();
    Dictionary entry = _reg_textures()[tex_id];
    return entry["t2drd"];
}
    #pragma endregion

#pragma endregion


