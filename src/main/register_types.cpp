#include "register_types.hpp"

#include "mn_common.hpp"     //LOG / JOIN / 断言宏

//GPGPU
#include "gpgpu.hpp"
#include "gpu_pipeline.hpp"
#include "gpgpu_worker.hpp"
#include "gpgpu_task.hpp"
using namespace godot;


// 初始化GDExtension
void initialize_module(ModuleInitializationLevel p_level){
    	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
    LOG_SET_LEVEL(LOG_LEVEL_DEBUG);

    // 在这里把类注册到 Godot 中
    ClassDB::register_class<GPGPU>();         //管理主渲染设备 管理worker线程
    ClassDB::register_class<GPUPipeline>();   //用户操作的管线对象

    GPGPU::init_registry();   // 初始化纹理注册表
    
}

void uninitialize_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) return;
    GPGPU::cleanup_worker();         //清理线程防止主线程退出后,子线程还未退出
    GPGPU::cleanup_registry();       //清空纹理注册表
}


// 入口函数 固定写法
extern "C" {
GDExtensionBool GDE_EXPORT more_nodes_init(GDExtensionInterfaceGetProcAddress p_get_proc_address,
    GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
    ::godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

    init_obj.register_initializer(initialize_module);
    init_obj.register_terminator(uninitialize_module);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}
}




