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

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/basic/CharInfo.hpp>

namespace slua::parse
{
#define _Slua_COMMON_KWS \
	"and", "break", "do", "else", "elseif", "end", "for", "function", \
	"goto", "if", "in", "local", "not", "or", "repeat", "return", \
	"then", "until", "while"
#define _Slua_KWS \
	/* freedom */\
	"continue", "where", "reloc", "loop", "raw", "ref", \
	/* future */\
	"share", "only", "box", "abstract", "become", "final", \
	"override", "typeof", "virtual", "unsized","global", \
	/* todos */\
	"copy", "move", "super",   \
	"generator", "gen",	"yield", "async", "await", "static", \
	/* documented */\
	"it", "to", "as", "of", "fn", "ex", "dyn", "let", "try", "use", "mut", "mod" \
	"also","case", "drop", "enum", "impl","with", "type","safe", "const", \
	"alloc", "macro", "match", "catch", "throw","trait", "union", \
	"axiom","unsafe","struct", "module", "extern", "comptime"

	inline const std::unordered_set<std::string> RESERVED_KEYWORDS = {
		"false", "nil", _Slua_COMMON_KWS, "true"
	};
	inline const std::unordered_set<std::string> RESERVED_KEYWORDS_SLUA = {
		_Slua_COMMON_KWS,
		_Slua_KWS,

		//Conditional
		"self", "Self", "crate",
	};
	inline const std::unordered_set<std::string> RESERVED_KEYWORDS_SLUA_MP_START = {
		_Slua_COMMON_KWS,
		_Slua_KWS
	};
#undef _LUA_KWS
#undef _Slua_COMMON_KWS

	template<bool forMpStart>
	inline bool isNameInvalid(AnyInput auto& in, const std::string& n)
	{
		const std::unordered_set<std::string>* checkSet = &RESERVED_KEYWORDS;

		if constexpr (in.settings() & sluaSyn)
		{
			if constexpr (forMpStart)
				checkSet = &RESERVED_KEYWORDS_SLUA_MP_START;
			else
				checkSet = &RESERVED_KEYWORDS_SLUA;
		}

		// Check if the resulting string is a reserved keyword
		if (checkSet->find(n) != checkSet->end())
			return true;
		return false;
	}

	template<bool forMpStart=false>
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

		// Ensure the first character is valid (a letter or underscore)
		if (!isValidNameStartChar(firstChar))
		{
			if (allowError)
				return "";
			throw UnexpectedCharacterError("Invalid identifier/name start: must begin with a letter or underscore" + errorLocStr(in));
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
		if (isNameInvalid<forMpStart>(in, res))
		{
			if (allowError)
				return "";
			throw ReservedNameError("Invalid identifier: matches a reserved keyword" + errorLocStr(in));
		}

		return res;
	}
	//No space skip!
	//Returns SIZE_MAX, on non name inputs
	//Otherwise, returns last peek() idx that returns a part of the name
	template<bool forMpStart = false>
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
		while (in)
		{
			const uint8_t ch = in.peekAt(i);// Starts at 1
			if (!isValidNameChar(ch))
				break; // Stop when a non-identifier character is found

			i++;

			res += ch;
			continue;
		}
		// Check if the resulting string is a reserved keyword
		if (isNameInvalid<forMpStart>(in, res))
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