#include "gpgpu_worker.hpp"

#include "mn_common.hpp"
#include "gpgpu_common.hpp"

#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

GPGPUWorker::GPGPUWorker() {
    _thread.instantiate();  //线程
    _mutex.instantiate();   //锁
    _sem.instantiate();     //唤醒量
}

GPGPUWorker::~GPGPUWorker() {
    stop();
}


#pragma region 主线程调用

// 主线程调用投任务
void GPGPUWorker::push_task(const Ref<GPUTask> &task) {
    _mutex->lock();
    _task_queue.append(task);
    int n = _task_queue.size();
    _mutex->unlock();
    _sem->post();
    LOG("投任务，队列长度=", n);
}


// 主线程调用查结果数量
int GPGPUWorker::get_result_count() {    //获取任务完成结果的 数量
    _mutex->lock();
    int c = _result_queue.size();
    _mutex->unlock();
    return c;
}

//主线程调用取结果
Dictionary GPGPUWorker::pop_result() {    //获取 第一个任务完成 项
    _mutex->lock();
    if (_result_queue.is_empty()) {
        _mutex->unlock();
        return Dictionary();
    }
    Dictionary r = _result_queue[0];
    _result_queue.remove_at(0);
    _mutex->unlock();
    return r;
}

#pragma endregion




#pragma region 线程启停
//启动线程
void GPGPUWorker::start() {
    if (_running) return;   //检测线程是否启用,启用就跳过
    _running = true;        //启用线程
    _thread->start(callable_mp(this, &GPGPUWorker::_worker_loop));   //线程开始循环
}

//停止线程
void GPGPUWorker::stop() {
    if (!_running) return;  //检测线程是否停用,停用就跳过
    _running = false;

    _sem->post();
    if (_thread->is_started()) {
        _thread->wait_to_finish();
    }
}

#pragma endregion



// 从主设备 handle 建本地 RD 纹理引用的RID
RID GPGPUWorker::_ensure_local_texture(uint64_t handle, const String &fmt,
                                       int w, int h, uint64_t usage_bits) {
    if (_local_rd == nullptr) {
        LOG_W("[Worker] _ensure_local_texture: 本地设备未就绪");
        return RID();
    }
    if (handle == 0) {
        LOG_W("[Worker] _ensure_local_texture: handle=0");
        return RID();
    }

    RenderingDevice::DataFormat rd_fmt = _glsl_format_to_rd(fmt);

    RID local_rid = _local_rd->texture_create_from_extension(
        RenderingDevice::TEXTURE_TYPE_2D,
        rd_fmt,
        RenderingDevice::TEXTURE_SAMPLES_1,   // 采样数枚举
        usage_bits,
        handle,
        w, h, 1, 1
    );

    if (!local_rid.is_valid()) {
        LOG_W("[Worker] _ensure_local_texture: texture_create_from_extension 失败",
              " fmt=", fmt, " ", w, "x", h);
        return RID();
    }
    return local_rid;
}


#pragma region 线程主循环
void GPGPUWorker::_worker_loop() {  //主循环
    LOG("[Worker] 开始工作");

    // 创建本地设备（整个生命周期就这一个）
    _local_rd = RenderingServer::get_singleton()->create_local_rendering_device();
    if (_local_rd == nullptr) {
        LOG_E("[Worker] 本地渲染设备创建失败");
        return;
    }
    LOG_I("[Worker] 本地渲染设备已创建 id:", (uint64_t)_local_rd);

    //每次 _sem->wait() 被唤醒，就走一遍 while 循环体，然后回到 wait() 再次阻塞
    while (true) {

        _sem->wait();   
        if (!_running) break;   //如果不需要线程运行了,就跳出循环重新堵塞


        // 取任务        
        _mutex->lock();
        if (_task_queue.is_empty()) {
            _mutex->unlock();
            continue;
        }
        Ref<GPUTask> task = _task_queue[0];
        _task_queue.remove_at(0);  //取走第一项任务
        _mutex->unlock();

        LOG_I("[Worker] 收到任务:", task->task_type,
            " pid=", task->pipeline_id,
            " iter=", task->iterations,
            " pass数=", task->passes.size());

        #pragma region 清理任务
        // 处理 cleanup 任务
        if (task->task_type == "cleanup") {
            int64_t pid = task->pipeline_id;
            LOG("[Worker] 清理 GPU管线缓存, pid=", pid);

            if (_pipeline_caches.has(pid)) {
                Dictionary cache = _pipeline_caches[pid];

                // 释放 display 本地 RID(如果启用了)
                if (cache.has("display_rids")) {
                    Array display_rids = cache["display_rids"];
                    for (int i = 0; i < display_rids.size(); i++) {
                        RID r = display_rids[i];
                        if (r.is_valid()) _local_rd->free_rid(r);
                    }
                    LOG("[Worker] 已释放 ", display_rids.size(), " 张 display 本地 RID");
                }

                // 释放纹理引用
                Dictionary tex_refs = cache["texture_ref_rids"];
                Array k1 = tex_refs.keys();
                for (int i = 0; i < k1.size(); i++) {
                    RID r = tex_refs[k1[i]];
                    if (r.is_valid()) _local_rd->free_rid(r);
                }


                // 释放 storage buffer
                Dictionary sb_rids = cache["storage_buffer_rids"];
                Array k2 = sb_rids.keys();
                for (int i = 0; i < k2.size(); i++) {
                    RID r = sb_rids[k2[i]];
                    if (r.is_valid()) _local_rd->free_rid(r);
                }

                // 释放 uniform buffer
                Dictionary ub_rids = cache["uniform_buffer_rids"];
                Array k3 = ub_rids.keys();
                for (int i = 0; i < k3.size(); i++) {
                    RID r = ub_rids[k3[i]];
                    if (r.is_valid()) _local_rd->free_rid(r);
                }

                // 每 pass 独立资源
                Array pass_caches = cache["pass_caches"];
                for (int pi = 0; pi < pass_caches.size(); pi++) {
                    Dictionary pc = pass_caches[pi];
                    
                    //释放unifrom_set
                    Dictionary us_rids = pc["uniform_set_rids"];
                    Array k4 = us_rids.keys();
                    for (int i = 0; i < k4.size(); i++) {
                        RID r = us_rids[k4[i]];
                        if (r.is_valid() && _local_rd->uniform_set_is_valid(r)) {
                            _local_rd->free_rid(r);
                        }
                    }
                    //释放pass管线
                    RID pipeline_rid = pc["pipeline_rid"];
                    if (pipeline_rid.is_valid()) _local_rd->free_rid(pipeline_rid);
                    //释放pass着色器
                    RID shader_rid = pc["shader_rid"];
                    if (shader_rid.is_valid()) _local_rd->free_rid(shader_rid);
                }

                _pipeline_caches.erase(pid);
            }
            
            continue;   // 处理下一个任务
        }
        #pragma endregion

        #pragma region 读回存储buffer 
        // 处理 readback_buffer 任务
        if (task->task_type == "readback_buffer") {
            int64_t rb_pid = task->pipeline_id;
            String rb_name = task->readback_buffer_name;

            Variant rb_data = PackedByteArray();   // 默认空
            String  rb_error = "";                 // 默认无错误

            if (_pipeline_caches.has(rb_pid)) {
                Dictionary rb_cache = _pipeline_caches[rb_pid];
                Dictionary sb_rids = rb_cache["storage_buffer_rids"];
                Dictionary sb_types = rb_cache["storage_buffer_types"];

                if (sb_rids.has(rb_name)) {
                    RID rid = sb_rids[rb_name];
                    PackedByteArray raw = _local_rd->buffer_get_data(rid, 0, 0);
                    String type = sb_types.has(rb_name) ? (String)sb_types[rb_name] : "byte";

                    if (type == "float32") {
                        rb_data = raw.to_float32_array();
                    } else if (type == "int32") {
                        rb_data = raw.to_int32_array();
                    } else {
                        rb_data = raw;
                    }
                    LOG("[Worker] readback buffer: ", rb_name, " 字节=", raw.size(), " type=", type);
                } else {
                    LOG_W("[Worker] readback buffer 未找到: ", rb_name);
                    rb_data = PackedByteArray();
                }
            } else {
                LOG_W("[Worker] readback buffer: pipeline cache 不存在 pid=", rb_pid);
                rb_data = PackedByteArray();
            }

            {
            _mutex->lock();
            Dictionary r;
            r["pipeline_id"] = rb_pid;
            r["signal"]      = "storage_buffer_ready";
            Array rb_args;
            rb_args.append(rb_name);
            rb_args.append(rb_data);
            r["args"] = rb_args;
            _result_queue.append(r);
            _mutex->unlock();
            }
            continue;
        }

        #pragma endregion


        #pragma region create_display 任务
        // worker 收到 create_display：
        //   1. 确保 pipeline cache 存在（create_display 可能在第一次 compute 前就到）
        //   2. 用 _ensure_local_texture 建 3 个本地 RID
        //   3. 存进 cache，供后续 compute 阶段 texture_copy 使用
        if (task->task_type == "create_display") {
            int64_t dp_pid = task->pipeline_id;

            LOG_I("[Worker] create_display 收到: pid=", dp_pid,
                " handles=", task->display_handles.size(),
                " source=", task->display_source_a, "<->", task->display_source_b,
                " ", task->display_width, "x", task->display_height);

            // 1. 确保 cache 存在
            if (!_pipeline_caches.has(dp_pid)) {
                Dictionary c;
                c["texture_ref_rids"]      = Dictionary();
                c["texture_handle_snap"]   = Dictionary();
                c["storage_buffer_types"]  = Dictionary();
                c["storage_buffer_rids"]   = Dictionary();
                c["uniform_buffer_rids"]   = Dictionary();
                c["pass_caches"]           = Array();
                c["pair_flip_count"]       = Dictionary();
                _pipeline_caches[dp_pid] = c;
                LOG_I("[Worker] create_display 时 cache 不存在，先建空 cache");
            }
            Dictionary cache = _pipeline_caches[dp_pid];

            // 2. 建 3 个本地 RID
            //   worker 侧作 texture_copy 目标：必须 CAN_COPY_TO_BIT
            //   其他 bit（SAMPLING / CAN_COPY_FROM）保留以便未来扩展；不要求 ⊆ 主设备也行，
            //   但为保险起见直接对齐主线程 _create_rd_texture 的完整集合
            uint64_t display_usage =
                RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT
                | RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT
                | RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

            Array handles = task->display_handles;
            String fmt_str = task->display_format;
            int w = task->display_width;
            int h = task->display_height;

            Array display_rids;
            bool build_ok = true;
            for (int i = 0; i < handles.size(); i++) {
                uint64_t handle = (uint64_t)(int64_t)handles[i];
                RID local_rid = _ensure_local_texture(handle, fmt_str, w, h, display_usage);
                if (!local_rid.is_valid()) {
                    LOG_E("[Worker] create_display: 建 display[", i, "] 本地 RID 失败");
                    build_ok = false;
                    break;
                }
                display_rids.append(local_rid);
                LOG_I("[Worker] display[", i, "] 本地 RID=", local_rid.get_id());
            }

            if (!build_ok) {
                // 回滚已建的
                for (int i = 0; i < display_rids.size(); i++) {
                    RID r = display_rids[i];
                    if (r.is_valid()) _local_rd->free_rid(r);
                }
                // 不存 cache，回 result 让主线程不卡
                _mutex->lock();
                Dictionary r;
                r["pipeline_id"] = dp_pid;
                r["signal"]      = "task_completed";
                r["args"]        = Array();
                r["error"]       = "create_display_failed";
                _result_queue.append(r);
                _mutex->unlock();
                continue;
            }

            // 3. 存进 cache
            cache["display_rids"]     = display_rids;
            cache["display_source_a"] = task->display_source_a;
            cache["display_source_b"] = task->display_source_b;
            cache["display_written"]  = 0;
            cache["display_w"]        = w;      
            cache["display_h"]        = h;      
            _pipeline_caches[dp_pid]  = cache;

            LOG_I("[Worker] create_display 完成,3 张 display 本地 RID 已建");

            // 4. 回 result
            {
                _mutex->lock();
                Dictionary r;
                r["pipeline_id"] = dp_pid;
                r["signal"]      = "task_completed";
                r["args"]        = Array();
                _result_queue.append(r);
                _mutex->unlock();
            }
            continue;
        }
        #pragma endregion




        #pragma region 计算任务
        // 0. 查/建 pipeline cache
        //如果是首次计算需要构建管线缓存
        int64_t pid = task->pipeline_id;
        if (!_pipeline_caches.has(pid)) {
            Dictionary c;


            c["texture_ref_rids"]      = Dictionary();   // name → RID   本地设备RID (通过共享纹理的引用)
            c["texture_handle_snap"]   = Dictionary();   // name → int64   主设备纹理的原生句柄

            c["storage_buffer_types"] = Dictionary();   // name → "float32"/"int32"/"byte"   记录存储的原始类型(用于读回)
            c["storage_buffer_rids"]    = Dictionary();  // name → RID                       记录存储buffer_本地设备RID

            c["uniform_buffer_rids"]    = Dictionary();

            c["pass_caches"]           = Array();         //pass缓存
            c["pair_flip_count"]           = Dictionary();   // 乒乓相位缓存
            _pipeline_caches[pid] = c;
            LOG_I("[Worker] 首次计算,构建管线缓存 , pid=", pid);
        }

        Dictionary cache = _pipeline_caches[pid];   //本管线的缓存
        // ---- 收集所有 pass 用到的纹理（并集）----
        //   同一个纹理可能被多个 pass 引用，只需创建一份本地 RID
        Dictionary all_textures;
        for (int pi = 0; pi < task->passes.size(); pi++) {
            Dictionary pt = task->passes[pi];
            Array texs = pt["textures"];
            for (int i = 0; i < texs.size(); i++) {
                Dictionary ti = texs[i];
                String nm = ti["name"];
                if (!all_textures.has(nm)) {
                    all_textures[nm] = ti;
                } else {
                    // 已存在——如果新版有 init_data 而旧版没有，则用新版覆盖
                    Dictionary exist = all_textures[nm];
                    if (ti.has("init_data") && !exist.has("init_data")) {
                        exist["init_data"] = ti["init_data"];
                        all_textures[nm] = exist;
                    }
                }
            }
        }

        #pragma region 共享纹理
        // 1. 引用/复用 共享纹理
        Dictionary tex_ref_rids   = cache["texture_ref_rids"];     //从缓存中取出上次创建的 纹理RID和主设备原生句柄
        Dictionary tex_handle_snap = cache["texture_handle_snap"];
        Dictionary local_texture_rids;   // 本次 name → RID

        Array all_tex_names = all_textures.keys();
        for (int i = 0; i < all_tex_names.size(); i++) {   //遍历所有 pass 用到的纹理并集
            String name = all_tex_names[i];
            Dictionary ti = all_textures[name];
            uint64_t handle = ti["handle"];    //本次的原生句柄
            String fmt      = ti["format"];
            int w           = ti["width"];
            int h           = ti["height"];

            int64_t cached_h = 0;    //cached_h是指该纹理是否是首次引用
            if (tex_handle_snap.has(name)) {
                cached_h = (int64_t)tex_handle_snap[name];   //如果上次创建了该名称的纹理那就先拿着
            }

            RID local_rid;

            if (cached_h == (int64_t)handle && tex_ref_rids.has(name)) {   //显存引用没变  且 有缓存本地RID
                //复用分支
                local_rid = tex_ref_rids[name];
                LOG_I("[Worker] 复用纹理引用: ", name, " rid=", local_rid.get_id());

            } else { // 没得复用那只能重建了
                //重建分支
                if (tex_ref_rids.has(name)) {
                    RID old = tex_ref_rids[name];
                    if (old.is_valid()) {
                        _local_rd->free_rid(old);
                        LOG("[Worker] 释放旧纹理引用: ", name);
                    }
                }

                // 共享纹理（state/velocity/hardness/cloud）在 worker 上是 compute 用：
                //   STORAGE（可读写 image2D 绑定）+ CAN_UPDATE（init_data 写入）
                //   + CAN_COPY_FROM（读回 / 后续可能 copy 源）+ SAMPLING
                uint64_t usage =
                    RenderingDevice::TEXTURE_USAGE_STORAGE_BIT
                    | RenderingDevice::TEXTURE_USAGE_CAN_UPDATE_BIT
                    | RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT
                    | RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT;

                local_rid = _ensure_local_texture(handle, fmt, w, h, usage);
                if (!local_rid.is_valid()) {
                    LOG_E("[Worker] 建纹理引用失败: ", name);
                    continue;   // 跳过这张，后续可能数据缺失但不会崩
                }

                // worker 首次创建纹理引用时，把主线程传来的初始数据写进显存
                if (ti.has("init_data")) {
                    PackedByteArray init_data = ti["init_data"];
                    if (init_data.size() > 0) {
                        Error err = _local_rd->texture_update(local_rid, 0, init_data);
                        if (err == OK) {
                            LOG_I("[Worker] 初始化纹理: ", name, " 字节=", init_data.size());
                        } else {
                            LOG_W("[Worker] 初始化纹理失败: ", name, " err=", (int)err);
                        }
                    }
                }
                //把本次的RID和原生句柄存到缓存里
                tex_ref_rids[name] = local_rid;
                tex_handle_snap[name] = (int64_t)handle;
                LOG("[Worker] 创建纹理引用: ", name, " rid=", local_rid.get_id());
            }

            local_texture_rids[name] = local_rid;
        }
        //把本次更新的写回
        cache["texture_ref_rids"] = tex_ref_rids;
        cache["texture_handle_snap"] = tex_handle_snap;
        #pragma endregion


        #pragma region 存储buffer
        // 收集所有 pass 用到的 storage buffer（并集）
        //   同一个 buffer 可能被多个 pass 引用，只要有一个 pass 带了 data 就保留
        Dictionary all_sb;
        for (int pi = 0; pi < task->passes.size(); pi++) {
            Array sbs = ((Dictionary)task->passes[pi])["storage_buffers"];
            for (int i = 0; i < sbs.size(); i++) {
                Dictionary sb = sbs[i];
                String nm = sb["name"];
                if (!all_sb.has(nm)) {
                    all_sb[nm] = sb;
                } else {
                    Dictionary exist = all_sb[nm];
                    if (sb.has("data") && !exist.has("data")) {
                        exist["type"] = sb["type"];
                        exist["data"] = sb["data"];
                        all_sb[nm] = exist;
                    }
                }
            }
        }
        // 2. 创建/复用 storage buffer（byte 内容对比）
        Dictionary sb_rids = cache["storage_buffer_rids"];
        Dictionary sb_types = cache["storage_buffer_types"];
        Dictionary local_storage_buffers;   // name → RID

        Array all_sb_names = all_sb.keys();
        for (int i = 0; i < all_sb_names.size(); i++) {
            String name = all_sb_names[i];
            Dictionary sb = all_sb[name];

            // 没收到 data → 复用 cache 里的 RID
            if (!sb.has("data")) {//复用分支
                if (sb_rids.has(name)) {
                    RID cached_rid = sb_rids[name];
                    local_storage_buffers[name] = cached_rid;
                    LOG("[Worker] 复用 storage buffer: ", name, " rid=", cached_rid.get_id());
                } else {
                    LOG_W("[Worker] storage buffer 需要 data 但 cache 里没有: ", name);
                }
                continue;
            }else{//重建分支
                String type = sb["type"];
                Variant data_v = sb["data"];

                PackedByteArray bytes;
                if (type == "float32")      bytes = ((PackedFloat32Array)data_v).to_byte_array();
                else if (type == "int32")   bytes = ((PackedInt32Array)data_v).to_byte_array();
                else if (type == "byte")    bytes = (PackedByteArray)data_v;
                else { LOG_W("[Worker] 未知 sbuf type: ", type); continue; }

                // 释放旧的
                if (sb_rids.has(name)) {
                    RID old = sb_rids[name];
                    if (old.is_valid()) _local_rd->free_rid(old);
                }
                RID buf_rid = _local_rd->storage_buffer_create(bytes.size(), bytes);
                sb_rids[name] = buf_rid;
                local_storage_buffers[name] = buf_rid;
                sb_types[name] = type;
                LOG("[Worker] 创建 storage buffer: ", name, " rid=", buf_rid.get_id());
            }
        }
        cache["storage_buffer_rids"] = sb_rids;
        cache["storage_buffer_types"] = sb_types;
        #pragma endregion

        #pragma region 只读buffer
        // ---- 收集所有 pass 用到的 uniform buffer（并集）----
        Dictionary all_ub;
        for (int pi = 0; pi < task->passes.size(); pi++) {
            Array ubs = ((Dictionary)task->passes[pi])["uniform_buffers"];
            for (int i = 0; i < ubs.size(); i++) {
                Dictionary ub = ubs[i];
                String nm = ub["name"];
                if (!all_ub.has(nm)) {
                    all_ub[nm] = ub;
                } else {
                    Dictionary exist = all_ub[nm];
                    if (ub.has("data") && !exist.has("data")) {
                        exist["type"] = ub["type"];
                        exist["data"] = ub["data"];
                        all_ub[nm] = exist;
                    }
                }
            }
        }
        // 3. 创建/复用 uniform buffer
        Dictionary ub_rids = cache["uniform_buffer_rids"];

        Dictionary local_uniform_buffers;
        Array all_ub_names = all_ub.keys();
        for (int i = 0; i < all_ub_names.size(); i++) {
            String name = all_ub_names[i];
            Dictionary ub = all_ub[name];

            if (!ub.has("data")) {//复用分支
                if (ub_rids.has(name)) {
                    RID cached_rid = ub_rids[name];
                    local_uniform_buffers[name] = cached_rid;
                    LOG("[Worker] 复用 uniform buffer: ", name, " rid=", cached_rid.get_id());
                } else {
                    LOG_W("[Worker] uniform buffer 需要 data 但 cache 里没有: ", name);
                }
                continue;
            }else{//重建分支
                String type = ub["type"];
                Variant data_v = ub["data"];

                PackedByteArray bytes;
                if (type == "float32")      bytes = ((PackedFloat32Array)data_v).to_byte_array();
                else if (type == "int32")   bytes = ((PackedInt32Array)data_v).to_byte_array();
                else if (type == "byte")    bytes = (PackedByteArray)data_v;
                else { LOG_W("[Worker] 未知 ubuf type: ", type); continue; }

                if (ub_rids.has(name)) {
                    RID old = ub_rids[name];
                    if (old.is_valid()) _local_rd->free_rid(old);
                }
                RID buf_rid = _local_rd->uniform_buffer_create(bytes.size(), bytes);
                ub_rids[name] = buf_rid;
                local_uniform_buffers[name] = buf_rid;
                LOG("[Worker] 创建 uniform buffer: ", name, " rid=", buf_rid.get_id());
            }
        }
        cache["uniform_buffer_rids"] = ub_rids;
        #pragma endregion

            #pragma region 每 pass 处理：pipeline / set_map / uniform_set
        // 每个 pass 独立处理——它有自己的 shader、binding 布局、uniform_set
        Array pass_caches = cache["pass_caches"];

        // 确保 pass_caches 数量 >= task->passes.size()
        while (pass_caches.size() < task->passes.size()) {
            Dictionary pc;
            pc["shader_path"]      = "";
            pc["shader_rid"]       = RID();
            pc["pipeline_rid"]     = RID();
            pc["spec_const_snap"]  = Dictionary();
            pc["uniform_set_rids"] = Dictionary();
            pc["uniform_set_snap"] = Dictionary();
            pass_caches.append(pc);
        }

        // all_uniform_sets 每个元素是 [flip0_list, flip1_list]
        Array all_uniform_sets;
        bool task_failed = false;
        Array pass_pingpong_usage;        // 每个 pass 引用的所有 pair（读+写）
        Array pass_pingpong_write;        // 每个 pass 写引用的 pair

        for (int pi = 0; pi < task->passes.size(); pi++) {
            Dictionary pt = task->passes[pi];
            Dictionary pc = pass_caches[pi];

            String pass_shader_path = pt["shader_path"];
            Dictionary pass_spec_consts = pt["spec_consts"];

            #pragma region 着色管线
            // 4. 加载/复用 shader + pipeline（按 pass 独立缓存）
            RID shader_rid   = pc["shader_rid"];
            RID pipeline_rid = pc["pipeline_rid"];
            Dictionary spec_snap = pc["spec_const_snap"];
            String cached_path = pc["shader_path"];

            bool need_rebuild_pipeline = !(pipeline_rid.is_valid()
                    && (cached_path == pass_shader_path)
                    && (spec_snap == pass_spec_consts));

            if (!need_rebuild_pipeline) {
                LOG("[Worker] pass ", pi, " 复用 pipeline");
            } else {
                if (pipeline_rid.is_valid()) _local_rd->free_rid(pipeline_rid);
                if (shader_rid.is_valid())   _local_rd->free_rid(shader_rid);
                pipeline_rid = RID();
                shader_rid   = RID();

                Ref<RDShaderFile> shader_file = ResourceLoader::get_singleton()->load(pass_shader_path);
                if (shader_file.is_null()) {
                    LOG_E("[Worker] pass ", pi, " 无法加载 shader: ", pass_shader_path);
                    cache["pass_caches"] = pass_caches;
                    _pipeline_caches[pid] = cache;
                    task_failed = true;
                    break;   
                }

                Ref<RDShaderSPIRV> spirv = shader_file->get_spirv();
                String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
                if (!err.is_empty()) {
                    LOG_E("[Worker] pass ", pi, " shader 编译错误: ", err);
                    cache["pass_caches"] = pass_caches;
                    _pipeline_caches[pid] = cache;
                    task_failed = true;
                    break;  
                }

                shader_rid = _local_rd->shader_create_from_spirv(spirv);

                TypedArray<Ref<RDPipelineSpecializationConstant>> spec_consts;
                Array spec_keys = pass_spec_consts.keys();
                for (int i = 0; i < spec_keys.size(); i++) {
                    int cid = spec_keys[i];
                    Dictionary spec_data = pass_spec_consts[cid];
                    String t = spec_data["type"];

                    Ref<RDPipelineSpecializationConstant> spec;
                    spec.instantiate();
                    spec->set_constant_id(cid);
                    if (t == "int")        spec->set_value((int)spec_data["value"]);
                    else if (t == "float") spec->set_value((float)spec_data["value"]);
                    else if (t == "bool")  spec->set_value((bool)spec_data["value"]);
                    spec_consts.append(spec);
                }

                pipeline_rid = _local_rd->compute_pipeline_create(shader_rid, spec_consts);

                pc["shader_path"]      = pass_shader_path;
                pc["shader_rid"]       = shader_rid;
                pc["pipeline_rid"]     = pipeline_rid;
                pc["spec_const_snap"]  = pass_spec_consts;

                LOG("[Worker] pass ", pi, " 创建 pipeline");
            }
            #pragma endregion

            #pragma region 组装 set_map
            // flip=1 时，pingpong 对里的名字要交换 RID
            auto resolve_texture_rid = [&](const String &name, bool flip) -> RID {
                RID rid = local_texture_rids[name];
                if (!flip) return rid;
                // 扫 pingpong 对，如果在里面就换成对面那张
                for (int k = 0; k < task->pingpong_pairs.size(); k++) {
                    Dictionary p = task->pingpong_pairs[k];
                    String pa = p["a_name"];
                    String pb = p["b_name"];
                    if (name == pa && local_texture_rids.has(pb)) return local_texture_rids[pb];
                    if (name == pb && local_texture_rids.has(pa)) return local_texture_rids[pa];
                }
                return rid;
            };

            // 按 flip 组装 set_map（textures 部分受 flip 影响，buffer 不受）
            auto build_set_map = [&](bool flip) -> Dictionary {
                Dictionary m;

                Array pass_textures = pt["textures"];
                for (int i = 0; i < pass_textures.size(); i++) {
                    Dictionary ti = pass_textures[i];
                    String name = ti["name"];
                    int set_num = ti["set"];
                    int binding = ti["binding"];
                    if (!local_texture_rids.has(name)) continue;
                    RID rid = resolve_texture_rid(name, flip);

                    if (!m.has(set_num)) {
                        TypedArray<Ref<RDUniform>> empty;
                        m[set_num] = empty;
                    }
                    Ref<RDUniform> u; u.instantiate();
                    u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_IMAGE);
                    u->set_binding(binding);
                    u->add_id(rid);
                    TypedArray<Ref<RDUniform>> arr = m[set_num];
                    arr.append(u);
                    m[set_num] = arr;
                }

                Array pass_sbs = pt["storage_buffers"];
                for (int i = 0; i < pass_sbs.size(); i++) {
                    Dictionary sb = pass_sbs[i];
                    String name = sb["name"];
                    int set_num = sb["set"];
                    int binding = sb["binding"];
                    if (!local_storage_buffers.has(name)) continue;
                    RID rid = local_storage_buffers[name];

                    if (!m.has(set_num)) {
                        TypedArray<Ref<RDUniform>> empty;
                        m[set_num] = empty;
                    }
                    Ref<RDUniform> u; u.instantiate();
                    u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
                    u->set_binding(binding);
                    u->add_id(rid);
                    TypedArray<Ref<RDUniform>> arr = m[set_num];
                    arr.append(u);
                    m[set_num] = arr;
                }

                Array pass_ubs = pt["uniform_buffers"];
                for (int i = 0; i < pass_ubs.size(); i++) {
                    Dictionary ub = pass_ubs[i];
                    String name = ub["name"];
                    int set_num = ub["set"];
                    int binding = ub["binding"];
                    if (!local_uniform_buffers.has(name)) continue;
                    RID rid = local_uniform_buffers[name];

                    if (!m.has(set_num)) {
                        TypedArray<Ref<RDUniform>> empty;
                        m[set_num] = empty;
                    }
                    Ref<RDUniform> u; u.instantiate();
                    u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER);
                    u->set_binding(binding);
                    u->add_id(rid);
                    TypedArray<Ref<RDUniform>> arr = m[set_num];
                    arr.append(u);
                    m[set_num] = arr;
                }
                return m;
            };
            #pragma endregion

            #pragma region uniform_set
            // 6. 用这个 pass 自己的 uniform_set 缓存
            Dictionary us_rids  = pc["uniform_set_rids"];
            Dictionary us_snaps = pc["uniform_set_snap"];

            // all_uniform_sets[pi] 里存两份：flip=0 的列表、flip=1 的列表
            Array uniform_sets_flip0;
            Array uniform_sets_flip1;

            for (int flip_i = 0; flip_i < 2; flip_i++) {
                bool flip = (flip_i == 1);
                Dictionary set_map = build_set_map(flip);
                Array set_nums = set_map.keys();
                Array out_list = flip ? uniform_sets_flip1 : uniform_sets_flip0;

                for (int i = 0; i < set_nums.size(); i++) {
                    int sn = set_nums[i];
                    TypedArray<Ref<RDUniform>> arr = set_map[sn];

                    // snap（指纹）
                    Array snap;
                    for (int j = 0; j < arr.size(); j++) {
                        Ref<RDUniform> u = arr[j];
                        Dictionary u_entry;
                        u_entry["binding"] = u->get_binding();
                        u_entry["type"]    = (int)u->get_uniform_type();
                        Array id_list;
                        Array ids = u->get_ids();
                        for (int k = 0; k < ids.size(); k++) {
                            id_list.append((int64_t)((RID)ids[k]).get_id());
                        }
                        u_entry["ids"] = id_list;
                        snap.append(u_entry);
                    }

                    // cache key 加 flip
                    String key = String::num_int64(sn) + "_" + String::num_int64(flip_i);

                    bool need_rebuild = need_rebuild_pipeline
                                    || !us_rids.has(key)
                                    || us_snaps[key] != snap;

                    RID us;
                    if (!need_rebuild) {
                        us = us_rids[key];
                        LOG("[Worker] pass ", pi, " 复用 uniform_set[", key, "]");
                    } else {
                        if (us_rids.has(key)) {
                            RID old = us_rids[key];
                            if (old.is_valid() && _local_rd->uniform_set_is_valid(old)) {
                                _local_rd->free_rid(old);
                            }
                        }
                        us = _local_rd->uniform_set_create(arr, shader_rid, sn);
                        us_rids[key]  = us;
                        us_snaps[key] = snap;
                        LOG("[Worker] pass ", pi, " 重建 uniform_set[", key, "]");
                    }

                    Dictionary entry;
                    entry["set"] = sn;
                    entry["rid"] = us;
                    out_list.append(entry);
                }
            }

            pc["uniform_set_rids"] = us_rids;
            pc["uniform_set_snap"] = us_snaps;
            pass_caches[pi] = pc;
            #pragma endregion
            // 收集这个 pass 引用了哪些 pingpong 对，其中哪些是写引用
            Array pass_tex_list = pt["textures"];
            Array used_pairs;
            Array write_pairs;

            for (int k = 0; k < task->pingpong_pairs.size(); k++) {
                Dictionary p = task->pingpong_pairs[k];
                String pa = p["a_name"];
                String pb = p["b_name"];

                bool is_used = false;
                bool is_write = false;
                for (int i = 0; i < pass_tex_list.size(); i++) {
                    Dictionary ti = pass_tex_list[i];
                    String tn = ti["name"];
                    if (tn != pa && tn != pb) continue;
                    is_used = true;
                    String access = ti.has("access") ? (String)ti["access"] : "";
                    if (access != "readonly") { is_write = true; break; }
                }

                if (is_used) {
                    used_pairs.append(pa);
                    if (is_write) write_pairs.append(pa);
                }
            }

            pass_pingpong_usage.append(used_pairs);
            pass_pingpong_write.append(write_pairs);

            if (used_pairs.size() > 0) {
                LOG("[Worker] pass ", pi, " 引用 pingpong: used=", used_pairs,
                    " write=", write_pairs);
            } else {
                LOG("[Worker] pass ", pi, " 无 pingpong");
            }
            // 存这个 pass 的 uniform_sets，供执行阶段用
            Array both;   // [flip0 列表, flip1 列表]
            both.append(uniform_sets_flip0);
            both.append(uniform_sets_flip1);
            all_uniform_sets.append(both);
        }

        // 写回 cache
        cache["pass_caches"] = pass_caches;
        _pipeline_caches[pid] = cache;
        #pragma endregion


        #pragma region 执行计算
        if (task_failed) {
            LOG_E("[Worker] 任务中止：有 pass 构建失败，pid=", pid);
            // 也要发 task_completed，不然主线程永远等不到信号
            {
                _mutex->lock();
                Dictionary r;
                r["pipeline_id"] = pid;
                r["signal"]      = "task_completed";
                r["args"]        = Array();
                r["error"]       = "compute_task_failed";
                _result_queue.append(r);
                _mutex->unlock();
            }
            continue;   // 回到 worker 主循环，处理下一个任务
        }

        // 每个 pair 的引用计数（跨 submit 保留）
        Dictionary pair_flip_count = cache["pair_flip_count"];   // ← 从 cache 拿，不新建

        int64_t cl = _local_rd->compute_list_begin();     //命令列表句柄  计算列表开始

        int iterations = task->iterations;
        int pass_count = task->passes.size();

        for (int iter = 0; iter < iterations; iter++) {
            for (int pi = 0; pi < pass_count; pi++) {
                Dictionary pt = task->passes[pi];
                Dictionary pc = pass_caches[pi];
                RID pipeline_rid = pc["pipeline_rid"];

                // 绑这个 pass 的 pipeline
                _local_rd->compute_list_bind_compute_pipeline(cl, pipeline_rid);

                // 按这个 pass 的 pingpong 相位选 uniform_set
                Array usage = pass_pingpong_usage[pi];
                bool need_flip = false;
                if (usage.size() > 0) {
                    String key = usage[0];
                    int cnt = pair_flip_count.has(key) ? (int)pair_flip_count[key] : 0;
                    need_flip = (cnt % 2 == 1);
                }

                Array both = all_uniform_sets[pi];
                Array uniform_sets = need_flip ? both[1] : both[0];

                for (int i = 0; i < uniform_sets.size(); i++) {
                    Dictionary entry = uniform_sets[i];
                    _local_rd->compute_list_bind_uniform_set(cl, entry["rid"], entry["set"]);
                }

                // 只对"写引用"的 pair 计数 +1
                Array writes = pass_pingpong_write[pi];
                for (int k = 0; k < writes.size(); k++) {
                    String key = writes[k];
                    int cnt = pair_flip_count.has(key) ? (int)pair_flip_count[key] : 0;
                    pair_flip_count[key] = cnt + 1;
                }

                // 这个 pass 的 push_const
                PackedByteArray pc_bytes = pt["push_const"];
                bool pass_has_push = pt.has("has_push_const") && (bool)pt["has_push_const"];
                if (pc_bytes.size() > 0) {
                    _local_rd->compute_list_set_push_constant(cl, pc_bytes, pc_bytes.size());
                }
                // 这个 pass 的 dispatch_groups
                Vector3i groups = pt["dispatch_groups"];
                _local_rd->compute_list_dispatch(cl, groups.x, groups.y, groups.z);

                // 不是最后一步 → barrier
                bool is_last = (iter == iterations - 1) && (pi == pass_count - 1);
                if (!is_last) _local_rd->compute_list_add_barrier(cl);
            }
        }

        _local_rd->compute_list_end();//命令队列结束

        // ---- display 三缓冲 copy（如果启用了）----
        //   在 submit 之前记录，和 compute 命令同一次提交，无额外 GPU 往返
        int  display_show_idx = -1;
        bool display_copied   = false;

        if (cache.has("display_rids")) {
            Array  display_rids = cache["display_rids"];
            String src_a        = cache["display_source_a"];
            String src_b        = cache["display_source_b"];
            int    written      = (int)cache["display_written"];
            int    dw           = (int)cache["display_w"];
            int    dh           = (int)cache["display_h"];

            // 找源的最新那张（a 或 b）
            int cnt = pair_flip_count.has(src_a) ? (int)pair_flip_count[src_a] : 0;
            String latest_name = (cnt % 2 == 1) ? src_b : src_a;

            Dictionary tex_refs = cache["texture_ref_rids"];
            if (tex_refs.has(latest_name) && display_rids.size() == 3 && dw > 0 && dh > 0) {
                RID src_rid = tex_refs[latest_name];
                int write_slot = written % 3;
                RID dst_rid = display_rids[write_slot];

                if (src_rid.is_valid() && dst_rid.is_valid()) {
                    // texture_copy：src 全尺寸 → display[write_slot]
                    // 参数：src, dst, src_pos, dst_pos, size, src_mip, dst_mip, src_layer, dst_layer
                    _local_rd->texture_copy(
                        src_rid, dst_rid,
                        Vector3(0, 0, 0),
                        Vector3(0, 0, 0),
                        Vector3(dw, dh, 1),
                        0, 0, 0, 0
                    );
                    display_copied = true;

                    int new_written = written + 1;
                    cache["display_written"] = new_written;

                    // show_idx：第 k 次写完后，显示第 k-2 次写的那张
                    //   new_written=1,2 → 显示还在攒，show=-1
                    //   new_written=3   → 显示第 2 次写的 display[1]
                    //   new_written=4   → 显示第 3 次写的 display[2]
                    //   new_written=5   → 显示第 4 次写的 display[0]
                    // 冷却时间：第 k-2 次写的槽位，到第 k 次时已经隔了 2 个 submit 周期，足够 GPU 队列执行完
                    if (new_written >= 3) {
                        display_show_idx = (new_written - 2) % 3;
                    }

                    LOG_I("[Worker] display copy: ", latest_name, " -> display[",
                        write_slot, "]  written=", new_written,
                        "  show_idx=", display_show_idx);
                }
            }
        }

        _local_rd->submit();   //递交任务
        _local_rd->sync();    //同步

        cache["pair_flip_count"] = pair_flip_count;
        _pipeline_caches[pid] = cache;
        // 通知主线程：这个 pipeline 的任务已经完成
        {
            _mutex->lock();
            Dictionary r;
            r["pipeline_id"] = pid;
            r["signal"]      = "task_completed";
            r["args"]        = Array();
            r["pair_flip_count"] = pair_flip_count;
            if (display_copied) {
                r["display_show_idx"] = display_show_idx;   // ← 新增
            }
            _result_queue.append(r);
            _mutex->unlock();
        }
        LOG("[Worker] compute 完成 pid=", pid);

    }
}
    #pragma endregion
#pragma endregion
