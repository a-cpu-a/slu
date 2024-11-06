/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <type_traits>

namespace slua
{
	//Internal, used bc, const T != T, etc
	template<typename T>
	using _RawType = std::remove_cv_t<std::remove_reference_t<T>>;


	//Internal, used bc, const T != T, etc
	template<typename T>
	using _ToLua = T;

	template<typename T>
	using ToLua = _ToLua<_RawType<T>>;




	template<typename T>
	concept LuaType = std::is_same_v<_RawType<T>, slua::ToLua<T>>;
	template<typename T>
	concept NonLuaType = !std::is_same_v<_RawType<T>, slua::ToLua<T>>;

	template<typename T>
	concept PushableLuaType = requires(slua::ToLua<T> t) {
		{ t.push(nullptr, t) } -> std::same_as<int>;
	};
}

// MUSN'T be inside a namespace !!!
// 
// the ... means template args
//
#define SLUA_MAP_TYPE(_WRAPPER,_NORMAL_TYPE,...) namespace slua { template<__VA_ARGS__>using _ToLua<_NORMAL_TYPE> = _WRAPPER; }