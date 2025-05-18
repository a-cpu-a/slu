/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <type_traits>

namespace slu
{
	//Internal, used bc, const T != T, etc
	template<typename T>
	using _RawType = std::remove_cv_t<std::remove_reference_t<T>>;


	//Internal, used bc, const T != T, etc
	template<typename T>
	struct _ToLua
	{
		using Type = T;
	};

	template<typename T>
	using ToLua = _ToLua<_RawType<T>>::Type;




	template<typename T>
	concept LuaType = std::is_same_v<_RawType<T>, slu::ToLua<T>>;
	template<typename T>
	concept NonLuaType = !std::is_same_v<_RawType<T>, slu::ToLua<T>>;

	template<typename T>
	concept Pushable = requires(slu::ToLua<T> t) {
		{ t.push(nullptr, t) } -> std::same_as<int>;
	};
}

//To let you use commas inside SLua_MAP_TYPE types
#define Slu_co ,

// MUST NOT be inside a namespace !!!
// 
// the ... is for template arguments
//
#define Slu_mapType(_NORMAL_TYPE,_WRAPPER,...) namespace slu { template<__VA_ARGS__>struct _ToLua<_NORMAL_TYPE> {using Type = _WRAPPER;}; }

// MUST NOT be inside a namespace !!!
// 
// like SLua_MAP_TYPE, except the wrapper can contain commas
//
#define Slu_mapType1(_NORMAL_TYPE,...) namespace slu { template<>struct _ToLua<_NORMAL_TYPE> {using Type = __VA_ARGS__;}; }