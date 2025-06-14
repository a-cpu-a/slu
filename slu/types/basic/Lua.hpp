﻿/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slu/Include.hpp>

#include <slu/types/Converter.hpp>
#include <slu/Utils.hpp>

namespace slu
{
	struct Nil {};

	template <typename T>
	struct Ret
	{
		Error errorMsg;
		slu::ToLua<T> val;
		bool errored = false;
		bool isNil = false;

		Ret(const T& val)
			:val(val)
		{}
		constexpr Ret(Nil)
			:isNil(true)
		{}
		constexpr Ret(const Error& message)
			:errored(true), errorMsg(message)
		{}
		constexpr Ret() = default;

		//Returns how many items were pushed to the stack, or negative in case of error
		static int push(lua_State* L, const Ret& data)
		{
			if (data.errored)
				throw data.errorMsg;

			if (!data.isNil)
				return slu::push(L, data.val);

			lua_pushnil(L);
			return 1;
		}
	};

	struct Void
	{
		static int push(lua_State* L, const Void& data) { return 0; }
		static constexpr const char* getName() { return "void"; }
	};

	//Like Void, except allows you to return errors too
	using VoidRet = Ret<Void>;
}