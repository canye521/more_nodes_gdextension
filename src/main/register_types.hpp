#pragma once

#include <godot_cpp/core/class_db.hpp>

using namespace godot;


// 初始化GDExtension
void initialize_module(ModuleInitializationLevel p_level);

// 反初始化GDExtension
void uninitialize_module(ModuleInitializationLevel p_level);
