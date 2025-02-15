/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cstdint>
#include <luaconf.h>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>

namespace sluaParse
{

	inline std::string errorLocStr(AnyInput auto& in) {
		return " " + in.fileName() + " (" LUACC_NUMBER + std::to_string(in.getLoc().line) + LUACC_DEFAULT "):" LUACC_NUMBER + std::to_string(in.getLoc().index);
	}

	struct UnicodeError : std::exception
	{
		std::string m;
		UnicodeError(const std::string& m) :m(m) {}
		const char* what() const { return m.c_str(); }
	};
	struct UnexpectedCharacterError : std::exception
	{
		std::string m;
		UnexpectedCharacterError(const std::string& m) :m(m) {}
		const char* what() const { return m.c_str(); }
	};
	struct UnexpectedFileEndError : std::exception
	{
		std::string m;
		UnexpectedFileEndError(const std::string& m) :m(m) {}
		const char* what() const { return m.c_str(); }
	};
	struct ReservedNameError : std::exception
	{
		std::string m;
		ReservedNameError(const std::string& m) :m(m) {}
		const char* what() const { return m.c_str(); }
	};
}