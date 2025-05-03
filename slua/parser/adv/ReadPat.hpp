/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>
#include <memory>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/ReadExprBase.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/errors/CharErrors.h>

namespace slua::parse
{
	template<AnyInput In>
	inline Pat readPat(In& in)
	{
		const char firstChar = in.peek();

		switch (firstChar)
		{
		case '.':
			if (checkReadToken(in, "..."))
			{
				//un op
			}
			[[fallthrough]];
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
			//readNumeral();
			break;
		case '{':
			//destructor
		case '-':
		case '!':
			//un op
		case '&':
		case '*':
			//type
		case '(':
			//exp or typeexp
			break;
		case '?':
		case '/':
			//type


		case '[':
			//string? type? array?
		case '"':
		case '\'':
			//str

		case 'a':
			//as?
		case 'd':
			//dyn?
		case 'f':
			//fn?
		case 'impl':
			//impl?
		case 'm':
			//mut?
		case 's':
			//struct? safe fn?
		case 't':
			//type? trait?
		case 'u':
			//union? unsafe fn?
		default:
			break;
		}
		throwExpectedPat(in);
	}
}