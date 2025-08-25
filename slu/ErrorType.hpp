/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <format>

namespace slu
{
	//throw inside a check to provide a custom message
	struct Error { 
		std::string msg;

		constexpr Error() = default;
		constexpr Error(const std::string& msg) :msg(msg) {}
		constexpr Error(std::string&& msg) :msg(std::move(msg)) {}

		template<class... Args>
		Error(const std::format_string<Args...> fmt, Args&&... fmtArgs)
			:msg(std::vformat(fmt.get(), std::make_format_args(fmtArgs...)))
		{}
	};
}
