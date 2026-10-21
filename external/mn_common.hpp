#pragma once

//工具函数
//LOG / JOIN / 断言宏

#include <godot_cpp/variant/string.hpp>                 // Godot 字符串
#include <godot_cpp/variant/utility_functions.hpp>      // Godot 工具函数

using namespace godot;


#pragma region LOG函数块

// 日志级别
enum LogLevel : int {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO  = 1,
    LOG_LEVEL_WARN  = 2,
    LOG_LEVEL_ERROR = 3,
    LOG_LEVEL_OFF   = 4,
};

inline int& LOG_MIN_LEVEL() {
    static int level = LOG_LEVEL_DEBUG;
    return level;
}

inline void LOG_SET_LEVEL(int level) { LOG_MIN_LEVEL() = level; }
inline void LOG_DISABLE()           { LOG_MIN_LEVEL() = LOG_LEVEL_OFF; }
inline void LOG_ENABLE()            { LOG_MIN_LEVEL() = LOG_LEVEL_DEBUG; }



// 参数转换
inline String LOG_ARG(const char* s) { return String::utf8(s); }
template<size_t N>
inline String LOG_ARG(const char (&s)[N]) { return String::utf8(s); }
inline String LOG_ARG(const String& s) { return s; }
template<typename T>
inline T LOG_ARG(const T& v) { return v; }



// 内部打印函数——带级别过滤
//   level < 全局最低级别 → 直接 return，不打印
template<typename... Args>
inline void _LOG_INTERNAL(int level, const Args&... args) {
    if (level < LOG_MIN_LEVEL()) return;
    UtilityFunctions::print(LOG_ARG(args)...);
}



// 各级别日志函数
//   现有 LOG(...) 调用点完全不用改——内部已经接入了级别判断
//   默认最低级别是 DEBUG → 和之前行为一致（全输出）
template<typename... Args>
inline void LOG(const Args&... args) {
    _LOG_INTERNAL(LOG_LEVEL_DEBUG, args...);
}

template<typename... Args>
inline void LOG_I(const Args&... args) {
    _LOG_INTERNAL(LOG_LEVEL_INFO, args...);
}

template<typename... Args>
inline void LOG_W(const Args&... args) {
    _LOG_INTERNAL(LOG_LEVEL_WARN, args...);
}

template<typename... Args>
inline void LOG_E(const Args&... args) {
    _LOG_INTERNAL(LOG_LEVEL_ERROR, args...);
}

#pragma endregion



#pragma region 断言宏

template<typename... Args>
inline String JOIN(Args... args) {
    String s;
    ((s += LOG_ARG(args)), ...);
    return s;
}

#define FAIL_IF(cond, ...) \
    ERR_FAIL_COND_MSG(cond, JOIN(__VA_ARGS__))

#define FAIL_IF_V(cond, ...) \
    ERR_FAIL_COND_V_MSG(cond, {}, JOIN(__VA_ARGS__))

#pragma endregion