/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cstdint>
#include <slu/ext/lua/luaconf.h>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>

namespace slu::parse
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

#define _Slu_MAKE_ERROR(_NAME) struct _NAME : ParseError \
	{ \
		using ParseError::ParseError;\
	}

	_Slu_MAKE_ERROR(UnicodeError);
	_Slu_MAKE_ERROR(UnexpectedKeywordError);
	_Slu_MAKE_ERROR(UnexpectedCharacterError);
	_Slu_MAKE_ERROR(UnexpectedFileEndError);
	_Slu_MAKE_ERROR(ReservedNameError);
	_Slu_MAKE_ERROR(ErrorWhileContext);
	_Slu_MAKE_ERROR(InternalError);

#undef _Slu_MAKE_ERROR
}