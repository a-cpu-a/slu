/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>

#include "SkipSpace.h"
#include "RequireToken.h"
#include "ReadName.h"

namespace sluaParse
{
	inline std::string readLabel(AnyInput auto& in)
	{
		//label ::= ‘::’ Name ‘::’

		skipSpace(in);
		requireToken(in, "::");

		const std::string res = readName(in);

		skipSpace(in);
		requireToken(in, "::");

		return res;
	}
	inline Statement readStatment(AnyInput auto& in)
	{
		/*
		stat ::=  ‘;’ |
		 varlist ‘=’ explist |
		 functioncall |
		 label |
		 break |
		 goto Name |
		 do block end |
		 while exp do block end |
		 repeat block until exp |
		 if exp then block {elseif exp then block} [else block] end |
		 for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end |
		 for namelist in explist do block end |
		 function funcname funcbody |
		 local function Name funcbody |
		 local attnamelist [‘=’ explist]
		*/
	}
	inline Block readBlock(AnyInput auto& in)
	{
		//block ::= {stat} [retstat]

		//0+ stat
		//0/1 return
	}

	struct ParsedFile
	{
		//TypeList types
		Block code;
	};

	inline ParsedFile parseFile(AnyInput auto& in)
	{
		return { readBlock(in) };
	}
}