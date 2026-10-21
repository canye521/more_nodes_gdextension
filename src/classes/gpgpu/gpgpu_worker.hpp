#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/semaphore.hpp>
#include <godot_cpp/classes/rendering_device.hpp>

#include "gpgpu_task.hpp"
#include <atomic>   

using namespace godot;
class GPGPUWorker : public RefCounted {


public:
    GPGPUWorker();
    ~GPGPUWorker();

    void start();                       // 启动后台线程
    void stop();                        // 停止后台线程


    //主线程调用
    void push_task(const Ref<GPUTask> &task);   //投递任务
    int  get_result_count();              //获取任务结果数量
    Dictionary pop_result();              //获取一项任务结果

private:

    Ref<Thread> _thread;          //后台线程对象
    Ref<CoreBind::Mutex> _mutex;  //互斥锁 防止数据冲突
    Ref<Semaphore> _sem;          //信号量 用来唤醒线程

    std::atomic<bool> _running{false};   //主线程是否要求后台线程停止工作
    RenderingDevice *_local_rd = nullptr;   // 后台线程持有的本地设备


    Array _task_queue;      // 待执行任务队列
    Array _result_queue;    // 已完成结果队列

    Dictionary _pipeline_caches;   // int64 pipeline_id → Dictionary（缓存）

    void _worker_loop();    // 后台线程主循环

    // 从主设备 handle 建本地 RD 纹理引用
    //   返回本地 RID；失败返回空 RID()
    RID _ensure_local_texture(uint64_t handle, const String &fmt,int w, int h, uint64_t usage_bits);
};