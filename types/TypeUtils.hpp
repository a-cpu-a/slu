/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <array>

namespace slua
{
	inline constexpr std::string cexpToString(size_t v)
	{
		std::string ret = "";

		while (v != 0)
		{
			ret += '0' + (v % 10);
			v /= 10;
		}

		if (ret.empty())
			return "0";

		return ret;
	}
	inline constexpr auto cexpStrToArray(auto cExprStrLambda, auto cExprLenLambda)
	{
		const std::string str = cExprStrLambda();
		std::array<char, cExprLenLambda() + 1> ret{};
		str.copy(ret.data(), str.size());
		return ret;
	}

}


// The proper way to use std::string for names
#define Slua_wrapGetStrName(_FUNC_NAME) \
	private: inline const static constexpr auto \
		name_buf = slua::cexpStrToArray( \
			[]() constexpr {return _FUNC_NAME(); }, \
			[]() constexpr {return _FUNC_NAME().size(); } \
		); \
	public: static constexpr const char* getName() { return name_buf.data(); }