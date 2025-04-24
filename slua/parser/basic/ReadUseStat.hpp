/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/adv/ReadName.hpp>

namespace slua::parse
{
	template<AnyInput In>
	inline bool readUseStat(In& in, const Position place, const ExportData exported)
	{
		if (checkReadTextToken(in, "use"))
		{
			StatementType::USE res{};
			res.exported = exported;

			res.base = readModPath(in);

			if (in.peek() == ':')
			{
				if (in.peekAt(1) == ':' && in.peekAt(2) == '*')
				{
					in.skip(3);
					res.useVariant = UseVariantType::EVERYTHING_INSIDE{};
				}
				else if (in.peekAt(1) == ':' && in.peekAt(2) == '{')
				{
					in.skip(3);
					UseVariantType::LIST_OF_STUFF list;
					list.push_back(in.genData.resolveUnknown(readName<true>(in)));
					while (checkReadToken(in, ","))
					{
						list.push_back(in.genData.resolveUnknown(readName<true>(in)));
					}
					requireToken(in, "}");
					res.useVariant = std::move(list);
				}
				else
				{// Neither, prob just no semicol
					res.useVariant = UseVariantType::IMPORT{};
				}
			}
			else
			{
				if (checkReadTextToken(in, "as"))
				{
					res.useVariant = UseVariantType::AS_NAME{ in.genData.resolveUnknown(readName(in))};
				}
				else
				{// Prob just no semicol
					res.useVariant = UseVariantType::IMPORT{};
				}
			}
			in.genData.addStat(place, std::move(res));
			return true;
		}
		return false;
	}
}