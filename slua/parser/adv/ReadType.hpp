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
			res.push_back(readModPath(in, readName<true>(in)));
			if (!checkReadToken(in, "+"))
				break;
		}
		return res;
	}
	template<AnyInput In>
	inline TypeSpecifiers readTypeSpecifiers(In& in)
	{
		TypeSpecifiers res{};

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
		while (in)
		{
			skipSpace(in);

			GcPtrLevel& lvl = res.gcPtrLevels.emplace_back();

			switch (in.peek())
			{
			case '#':
				lvl.isPtr = false;
				break;
			case '*':
				lvl.isPtr = true;
				break;
			default:
				goto exit;
			}
			in.skip();
			lvl.hasMut = checkReadTextToken(in, "mut");
		}
	exit:
		return res;
	}

	template<AnyInput In>
	inline TypeItem readTypeItem(In& in)
	{
		TypeItem ret;
		ret.prefix = readTypeSpecifiers(in);
		
		skipSpace(in);
		switch (in.peek())
		{
		case '['://Slice or array
		{
			in.skip();

			TypeObjType::SLICE_OR_ARRAY res{};
			res.ty = readType(in);

			while (checkReadToken(in, ";"))
			{
				res.size.emplace_back(readExpr(in, false));
			}

			requireToken(in, "]");

			ret.obj = std::move(res);
			return ret;
		}
		case '{':
		{
			in.skip();

			TypeObjType::TUPLE res{};

			if (in.peek() != '}')//Is not empty
			{
				res.emplace_back(readType(in));

				while (checkReadToken(in, ","))
				{
					res.emplace_back(readType(in));
				}
			}

			ret.obj = std::move(res);
			return ret;
		}
		case '(':
			in.skip();

			ret.obj = TypeObjType::TYPE(readType(in));

			requireToken(in, ")");
			return ret;
		case 'd':
			if (checkReadTextToken(in, "dyn"))
			{
				ret.obj = TypeObjType::TRAIT_COMBO(readTraitCombo(in), true);
				return ret;
			}
			break;
		case 'i':
			if (checkReadTextToken(in, "impl"))
			{
				ret.obj = TypeObjType::TRAIT_COMBO(readTraitCombo(in), false);
				return ret;
			}
			break;
		default:
			break;
		}
		//comptime var thing

		ret.obj = TypeObjType::COMPTIME_VAR_TYPE(readModPath(in, readName(in)));
		return ret;
	}
	template<AnyInput In>
	inline Type readType(In& in)
	{
		Type res{readTypeItem(in)};
		while (in)
		{
			if (!checkReadToken(in, "|"))
				break;
			res.emplace_back(readTypeItem(in));
		}
		return res;
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