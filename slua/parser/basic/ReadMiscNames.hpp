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
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/adv/ReadName.hpp>

namespace slua::parse
{
	template<AnyInput In>
	inline NameList<In> readNameList(In& in)
	{
		/*
			namelist ::= Name {‘,’ Name}
		*/
		NameList<In> ret{};
		ret.push_back(readName(in));

		while (checkReadToken(in, ","))
		{
			ret.push_back(readName(in));
		}
		return ret;
	}
	template<AnyInput In>
	inline AttribName<In> readAttribName(In& in)
	{
		AttribName<In> ret;
		ret.name = in.genData.resolveUnknown(readName(in));//Unknown, because this is only used in lua, and it doesnt matter there
		if (checkReadToken(in, "<"))
		{// attrib ::= [‘<’ Name ‘>’]
			ret.attrib = readName(in);
			requireToken(in, ">");
		}

		return ret;
	}
	template<AnyInput In>
	inline AttribNameList<In> readAttNameList(In& in)
	{
		/*
			attnamelist ::=  Name attrib {‘,’ Name attrib}
			attrib ::= [‘<’ Name ‘>’]
		*/
		AttribNameList<In> ret = { readAttribName(in) };

		while (checkReadToken(in, ","))
		{
			ret.push_back(readAttribName(in));
		}
		return ret;
	}
	inline std::string readFuncName(AnyInput auto& in)
	{
		// funcname ::= Name {‘.’ Name} [‘:’ Name]
		std::string ret = readName(in);

		while (checkReadToken(in, "."))
		{
			ret += "." + readName(in);
		}
		if (checkReadToken(in, ":"))
			ret += ":" + readName(in);

		return ret;
	}
}