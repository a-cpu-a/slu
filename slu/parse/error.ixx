module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>
#include <string>

export module slu.parse.error;

namespace slu::parse
{
	export struct ParseFailError : std::exception
	{
		const char* what() const
		{
			return "Failed to parse some slu/lua code.";
		}
	};

	export struct BasicParseError : std::exception
	{
		std::string m;
		BasicParseError(const std::string& m) : m(m) {}
		const char* what() const override
		{
			return m.c_str();
		}
	};
	export struct FailedRecoveryError : BasicParseError
	{
		using BasicParseError::BasicParseError;
	};

	export struct ParseError : BasicParseError
	{
		using BasicParseError::BasicParseError;
	};

#define MAKE_ERROR(_NAME)             \
	export struct _NAME : ParseError  \
	{                                 \
		using ParseError::ParseError; \
	}

	MAKE_ERROR(UnicodeError);
	MAKE_ERROR(UnexpectedKeywordError);
	MAKE_ERROR(UnexpectedCharacterError);
	MAKE_ERROR(UnexpectedFileEndError);
	MAKE_ERROR(ReservedNameError);
	MAKE_ERROR(ErrorWhileContext);
	MAKE_ERROR(InternalError);
} //namespace slu::parse