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
	struct ParseFailError : std::exception
	{
		const char* what() const { return "Failed to parse some (s)lua code."; }
	};

	struct BasicParseError : std::exception
	{
		std::string m;
		BasicParseError(const std::string& m) :m(m) {}
		const char* what() const { return m.c_str(); }
	};
	struct FailedRecoveryError : BasicParseError
	{
		using BasicParseError::BasicParseError;
	};

	struct ParseError : BasicParseError
	{
		using BasicParseError::BasicParseError;
	};

#define _Slua_MAKE_ERROR(_NAME) struct _NAME : ParseError \
	{ \
		using ParseError::ParseError;\
	}

	_Slua_MAKE_ERROR(UnicodeError);
	_Slua_MAKE_ERROR(UnexpectedKeywordError);
	_Slua_MAKE_ERROR(UnexpectedCharacterError);
	_Slua_MAKE_ERROR(UnexpectedFileEndError);
	_Slua_MAKE_ERROR(ReservedNameError);
	_Slua_MAKE_ERROR(ErrorWhileContext);

#undef _Slua_MAKE_ERROR
}