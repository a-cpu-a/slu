/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slua/Include.hpp>

#include <slua/types/Converter.hpp>
#include <slua/Utils.hpp>

namespace slua
{
	struct Nil {};
	struct Error { std::string msg; };

	template <typename T>
	struct Ret
	{
		Error errorMsg;
		slua::ToLua<T> value;
		bool errored = false;
		bool isNil = false;

		Ret(const T& _value)
			:value(_value)
		{}
		Ret(Nil)
			:isNil(true)
		{}
		Ret(const Error& message)
			:errored(true), errorMsg(message)
		{}
		Ret() {}

		int push(lua_State* L) const
		{
			if (errored)
				return slua::error(L, errorMsg.msg);

			if (!isNil)
				return slua::push(L, value);

			lua_pushnil(L);
			return 1;
		}
	};

	struct Void
	{
		static int push(lua_State* L, const Void& data) { return 0; }
		static constexpr const char* getName() { return "void"; }
	};

	using VoidRet = Ret<Void>;
}