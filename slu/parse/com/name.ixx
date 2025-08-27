module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>
#include <unordered_set>
#include <algorithm>
#include <format>

#include <slu/Ansi.hpp>
export module slu.parse.com.name;

import slu.char_info;
import slu.num;
import slu.ast.state;
import slu.parse.error;
import slu.parse.input;
import slu.parse.com.num;
import slu.parse.com.skip_space;
import slu.parse.com.token;

namespace slu::parse
{
#define _Slu_COMMON_KWS \
	"and", "break", "do", "else", "elseif", "end", "for", "function", \
	"global", "goto", "if", "in", "local", "or", "repeat", "return", \
	"then", "until", "while"

#define _Slu_KWS \
	_Slu_COMMON_KWS, \
	/* freedom */\
	"any", "has", "raw", "glob", "reloc", "concept", "nostride", \
	/* future */\
	"at", "of", "box", "auto", "case", "only", "with", "final", "become", \
	"typeof", "abstract", "comptime", "override", "unsized", "virtual", \
	/* todos */\
	"gen",	"copy", "move", "async", "await", \
	"yield", "static", "generator", \
	/* documented */\
	"as", "ex", "fn", "it", "dyn", "let", "mod", "mut", "try", "use" \
	"also", "drop", "enum", "impl", "loop", "safe", "alloc", "axiom", \
	"catch", "const", "defer", "macro", "match", "share", "throw", "trans", \
	"union", "where", "extern", "module", "struct", "unsafe", "continue"
#define _Slu_VERY_KWS "self", "crate", "super"
#define _Slu_MOSTLY_KWS _Slu_VERY_KWS, "Self"

	inline const std::unordered_set<std::string> RESERVED_KEYWORDS_SLU = {
		_Slu_KWS,

		//Conditional
		_Slu_MOSTLY_KWS, "trait",
	};
	inline const std::unordered_set<std::string> RESERVED_KEYWORDS_SLU_BOUND_VAR = {
		_Slu_KWS,
		_Slu_VERY_KWS, "trait"
	};
	inline const std::unordered_set<std::string> RESERVED_KEYWORDS_SLU_MP_START = {
		_Slu_KWS
	};
	inline const std::unordered_set<std::string> RESERVED_KEYWORDS_SLU_MP = {
		_Slu_KWS,
		_Slu_MOSTLY_KWS
	};

	export enum class NameCatagory
	{
		DEFAULT,
		MP_START,
		MP,
		BOUND_VAR
	};
	template<NameCatagory cata>
	bool isNameInvalid(AnyInput auto& in, const std::string& n)
	{
		const std::unordered_set<std::string>* checkSet = &RESERVED_KEYWORDS_SLU;

		if constexpr (cata == NameCatagory::MP_START)
			checkSet = &RESERVED_KEYWORDS_SLU_MP_START;
		else if constexpr (cata == NameCatagory::MP)
			checkSet = &RESERVED_KEYWORDS_SLU_MP;
		else if constexpr (cata == NameCatagory::BOUND_VAR)
			checkSet = &RESERVED_KEYWORDS_SLU_BOUND_VAR;

		// Check if the resulting string is a reserved keyword
		if (checkSet->find(n) != checkSet->end())
			return true;
		return false;
	}

	export template<NameCatagory cata =NameCatagory::DEFAULT,bool sluTuplable=false>
	std::string readName(AnyInput auto& in, const bool allowError = false)
	{
		/*
		Names (also called identifiers) in Lua can be any string
		of Latin letters, Arabic-Indic digits, and underscores,
		not beginning with a digit and not being a reserved word.
		*/
		skipSpace(in);

		if (!in)
			throw UnexpectedFileEndError("Expected identifier/name: but file ended" + errorLocStr(in));

		const uint8_t firstChar = in.peek();

		if constexpr(sluTuplable)
		{
			// Ensure the first character is valid (a letter, number or underscore)
			if (!isValidNameChar(firstChar))
			{
				if (allowError)
					return "";
				throw UnexpectedCharacterError(std::format(
					"Invalid identifier/"
					LC_integer
					"/name start: must begin with a letter, digit or underscore"
					"{}"
					, errorLocStr(in))
				);
			}
			if (isDigitChar(firstChar))
			{
				TupleName n = readNumeral<TupleName, false, false>(in, firstChar);
				return u128ToStr(n.lo, n.hi);
			}
		}
		else
		{
			// Ensure the first character is valid (a letter or underscore)
			if (!isValidNameStartChar(firstChar))
			{
				if (allowError)
					return "";
				throw UnexpectedCharacterError("Invalid identifier/name start: must begin with a letter or underscore" + errorLocStr(in));
			}
		}


		std::string res;
		res += firstChar;
		in.skip(); // Consume the first valid character

		// Consume subsequent characters (letters, digits, or underscores)
		while (in)
		{
			const uint8_t ch = in.peek();
			if (!isValidNameChar(ch))
				break; // Stop when a non-identifier character is found

			res += in.get();
			continue;
		}

		// Check if the resulting string is a reserved keyword
		if (isNameInvalid<cata>(in, res))
		{
			if (allowError)
				return "";
			throw ReservedNameError("Invalid identifier: matches a reserved keyword" + errorLocStr(in));
		}

		return res;
	}
	export template<NameCatagory cata = NameCatagory::DEFAULT>
	std::string readSluTuplableName(AnyInput auto& in, const bool allowError = false) {
		return readName<cata,true>(in, allowError);
	}

	//No space skip!
	//Returns SIZE_MAX, on non name inputs
	//Otherwise, returns last peek() idx that returns a part of the name
	export template<NameCatagory cata = NameCatagory::DEFAULT>
	size_t peekName(AnyInput auto& in,const size_t at = 0)
	{
		if (!in)
			return SIZE_MAX;

		const uint8_t firstChar = in.peekAt(at);

		// Ensure the first character is valid (a letter or underscore)
		if (!isValidNameStartChar(firstChar))
			return SIZE_MAX;

		std::string res;
		res += firstChar; // Consume the first valid character

		// Consume subsequent characters (letters, digits, or underscores)
		size_t i = 1+ at;
		while (!in.isOob(i))
		{
			const uint8_t ch = in.peekAt(i);// Starts at 1
			if (!isValidNameChar(ch))
				break; // Stop when a non-identifier character is found

			i++;

			res += ch;
			continue;
		}
		// Check if the resulting string is a reserved keyword
		if (isNameInvalid<cata>(in, res))
			return SIZE_MAX;

		return i;
	}
}