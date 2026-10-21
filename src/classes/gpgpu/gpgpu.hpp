#pragma once

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/texture2drd.hpp>
#include <godot_cpp/classes/rendering_device.hpp>

#include "mn_common.hpp" //LOG / JOIN / 断言宏
#include "gpu_pipeline.hpp"
#include "gpgpu_worker.hpp"
#include "gpgpu_task.hpp"


using namespace godot;

class GPGPU : public Object
{
	GDCLASS(GPGPU, Object)

protected:
	static void _bind_methods();

private:
	static RenderingDevice *_main_rd; // 主渲染设备

	static Ref<GPGPUWorker> _worker;		  // 工作线程
	static Ref<GPGPUWorker> _ensure_worker(); // 确保 后台工作线程存在

	// 主线程轮询
	static bool _poll_connected;	 // 是否已连接 process_frame 轮询（一次性，连上后每帧都跑）
	static void _on_process_frame(); // 每帧回调：检查 worker 结果队列，有结果就 emit 信号

public:
	GPGPU();
	~GPGPU();

	static RenderingDevice *get_main_rd(); // 获取渲染设备对象   //仅GDE内部使用

	static void push_task(const Ref<GPUTask> &task); // 管线需要投任务时调用 转发给后台线程
	static void cleanup_worker();					 // 仅GDE反初始化时调用

	// ---- 纹理注册表 ----
	// 注册纹理
	static int64_t register_texture(RID rid, uint64_t handle, const String &format, int w, int h, bool external);

	static void add_texture_ref(int64_t tex_id); // 添加引用
	static void release_texture(int64_t tex_id); // 移除引用
	// 查询
	static int64_t get_tex_id_by_rid(RID rid);				   // 通过RID获取tex_id
	static RID get_texture_rid(int64_t tex_id);				   // 通过tex_id -> RID
	static uint64_t get_texture_handle(int64_t tex_id);		   // 通过tex_id -> 原始纹理句柄
	static Dictionary get_texture_info(int64_t tex_id);		   // 通过tex_id -> format/width/height/external
	static Ref<Texture2DRD> get_texture_t2drd(int64_t tex_id); // 通过tex_id -> texture2drd

	// 初始化与移除
	static void init_registry();	// 仅GDE初始化时调用
	static void cleanup_registry(); // 释放注册表里所有  RID 仅GDE反初始化时调用

	// GDS调用
	static Ref<GPUPipeline> create_pipeline(const Variant &shader_paths); // 创建一个管线对象，并设置着色器路径

};
