/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>
#include <algorithm>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/RequireToken.hpp>
import slu.char_info;
#include <slu/parser/adv/ReadNumeral.hpp>

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
#undef _Slu_KWS
#undef _Slu_MOSTLY_KWS
#undef _Slu_VERY_KWS
#undef _Slu_COMMON_KWS
	enum class NameCatagory
	{
		DEFAULT,
		MP_START,
		MP,
		BOUND_VAR
	};
	template<NameCatagory cata>
	inline bool isNameInvalid(AnyInput auto& in, const std::string& n)
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

	template<NameCatagory cata =NameCatagory::DEFAULT,bool sluTuplable=false>
	inline std::string readName(AnyInput auto& in, const bool allowError = false)
	{
		/*
		Names (also called identifiers) in Lua can be any string
		of Latin letters, Arabic-Indic digits, and underscores,
		not beginning with a digit and not being a reserved word.

		The following keywords are reserved and cannot be used as names:

		 and       break     do        else      elseif    end
		 false     for       function  goto      if        in
		 local     nil       not       or        repeat    return
		 then      true      until     while
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
	template<NameCatagory cata = NameCatagory::DEFAULT>
	inline std::string readSluTuplableName(AnyInput auto& in, const bool allowError = false) {
		return readName<cata,true>(in, allowError);
	}

	//No space skip!
	//Returns SIZE_MAX, on non name inputs
	//Otherwise, returns last peek() idx that returns a part of the name
	template<NameCatagory cata = NameCatagory::DEFAULT>
	inline size_t peekName(AnyInput auto& in,const size_t at = 0)
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

	//uhhh, dont use?
	template<AnyInput In>
	inline NameList<In> readNames(In& in, const bool requires1 = true)
	{
		NameList<In> res;

		if (requires1)
			res.push_back(in.genData.resolveUnknown(readName(in)));

		bool skipComma = !requires1;//comma wont exist if the first one doesnt exist
		bool allowNameError = !requires1;//if the first one doesnt exist

		while (skipComma || checkReadToken(in, ','))
		{
			skipComma = false;// Only skip first comma

			const std::string str = readName(in, allowNameError);

			if (allowNameError && str.empty())
				return {};//no names

			res.push_back(in.genData.resolveUnknown(str));

			allowNameError = false;//not the first one anymore
		}
		return res;
	}
}