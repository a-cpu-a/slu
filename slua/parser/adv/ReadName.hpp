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

namespace sluaParse
{
#define _LUA_KWS \
	"and", "break", "do", "else", "elseif", "end", "false", "for", "function", \
	"goto", "if", "in", "local", "nil", "not", "or", "repeat", "return", \
	"then", "true", "until", "while"

	inline const std::unordered_set<std::string> RESERVED_KEYWORDS = {
		_LUA_KWS
	};
	inline const std::unordered_set<std::string> RESERVED_KEYWORDS_SLUA = {
		_LUA_KWS,

		//freedom
		"continue",
		"reloc",
		"where",
		"loop",
		"raw",
		"ref",

		//future
		"only",
		"box",
		"abstract",
		"become",
		"final",
		"override",
		"typeof",
		"virtual",
		"unsized",
		"const",

		//todos
		"enum",
		"struct",
		"trait",
		"union",
		"impl",
		"copy",
		"move",
		"super",
		"unsafe",
		"safe",
		"dyn",
		"generator",

		"yield",
		"async",
		"await",
		"static",

		//documented
		"to",
		"as",
		"of",
		"fn",
		"ex",
		"let",
		"try",
		"use",
		"mut",
		"gen",
		"drop",
		"case",
		"type",
		"self",
		"Self",
		"alloc",
		"macro",
		"match",
		"crate",
		"catch",
		"throw",
		"module",
		"comptime"
	};
#undef _LUA_KWS

	inline bool isNameInvalid(AnyInput auto& in, const std::string& n)
	{
		if constexpr (in.settings() & sluaSyn)
		{
			// Check if the resulting string is a reserved keyword
			if (RESERVED_KEYWORDS_SLUA.find(n) != RESERVED_KEYWORDS_SLUA.end())
				return true;
			return false;
		}

		// Check if the resulting string is a reserved keyword
		if (RESERVED_KEYWORDS.find(n) != RESERVED_KEYWORDS.end())
			return true;
		return false;
	}

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
		if (isNameInvalid(in, res))
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
	inline size_t peekName(AnyInput auto& in)
	{
		if (!in)
			return SIZE_MAX;


		const uint8_t firstChar = in.peek();

		// Ensure the first character is valid (a letter or underscore)
		if (!isValidNameStartChar(firstChar))
			return SIZE_MAX;


		std::string res;
		res += firstChar; // Consume the first valid character

		// Consume subsequent characters (letters, digits, or underscores)
		size_t i = 1;
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
		if (isNameInvalid(in, res))
			return SIZE_MAX;

		return i;
	}

	//uhhh, dont use?
	inline NameList readNames(AnyInput auto& in, const bool requires1 = true)
	{
		NameList res;

		if (requires1)
			res.push_back(readName(in));

		bool skipComma = !requires1;//comma wont exist if the first one doesnt exist
		bool allowNameError = !requires1;//if the first one doesnt exist

		while (skipComma || checkReadToken(in, ','))
		{
			skipComma = false;// Only skip first comma

			const std::string str = readName(in, allowNameError);

			if (allowNameError && str.empty())
				return {};//no names

			res.push_back(str);

			allowNameError = false;//not the first one anymore
		}
		return res;
	}
}