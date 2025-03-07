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
	struct ParseError : std::exception
	{
		std::string m;
		ParseError(const std::string& m) :m(m) {}
		const char* what() const { return m.c_str(); }
	};

	struct ParseFailError : std::exception
	{
		const char* what() const { return "Failed to parse some (s)lua code."; }
	};

#define _Slua_MAKE_ERROR(_NAME) struct _NAME : ParseError \
	{ \
		using ParseError::ParseError;\
	}

	_Slua_MAKE_ERROR(UnicodeError);
	_Slua_MAKE_ERROR(UnexpectedCharacterError);
	_Slua_MAKE_ERROR(UnexpectedFileEndError);
	_Slua_MAKE_ERROR(ReservedNameError);
	_Slua_MAKE_ERROR(ErrorWhileContext);

#undef _Slua_MAKE_ERROR
}