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

namespace sluaParse
{
	template<AnyInput In>
	inline TraitCombo readTraitCombo(In& in)
	{
		TraitCombo res;
		while (in)
		{
			res.push_back(parseModPath(in, readName<true>(in)));
			if (!checkReadToken(in, "+"))
				break;
		}
		return res;
	}
	template<AnyInput In>
	inline TypeSpecifiers readTypeSpecifiers(In& in)
	{
		TypeSpecifiers res{};
		skipSpace(in);
		if (checkReadToken(in, "#"))
			res.gc = true;

		while (in)
		{
			if (!checkReadToken(in, "&"))
				break;
			BorrowLevel& b = res.borrows.emplace_back();
			while (in)
			{
				if (!checkReadToken(in, "/"))
					break;
				b.lifetimes.push_back(readName(in));
			}

			b.hasMut = checkReadTextToken(in, "mut");
		}

		return res;
	}

	template<AnyInput In>
	inline Type readType(In& in)
	{
		return {};
	}

	/**
	 * @brief 
	 * @param res Output variable
	 * @return true, if you need to return
	 */
	template<AnyInput In>
	inline bool checkReadErrTypeQMark(In& in, ErrType& res)
	{
		if (in.peek() == '?')
		{
			res.isErr = true;
			if (in.peekAt(1) == '?')
			{// Its ?? or (type ??)
				in.skip(2);
				return true;
			}

			// Its (? type) or (type ? type)
			in.skip();
			res.err = readType();
			return true;
		}
		return false;
	}

	template<AnyInput In>
	inline ErrType readErrType(In& in)
	{
		ErrType res;

		if (checkReadErrTypeQMark(in, res))
			return res;

		// Its (type) or (type ??) or (type ? type)
		res.val = readType();

		checkReadErrTypeQMark(in, res);
		return res;
	}
}