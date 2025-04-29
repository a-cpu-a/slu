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
	inline bool readModStat(In& in, const Position place, const ExportData exported)
	{
		if (checkReadTextToken(in, "mod"))
		{
			std::string strName =
				exported
				? readName(in)
				: readName<true>(in);
			if (!exported)
			{
				if (strName == "self")
				{
					in.genData.addStat(place, StatementType::MOD_SELF{});
					return true;
				}
				if (strName == "crate")
				{
					in.genData.addStat(place, StatementType::MOD_CRATE{});
					return true;
				}
			}
			const MpItmId<In> modName = in.genData.resolveUnknown(strName);

			if (checkReadTextToken(in, "as"))
			{
				StatementType::MOD_DEF_INLINE<In> res{};
				res.exported = exported;
				res.name = modName;

				requireToken(in, "{");
				res.bl = readBlockNoStartCheck<false>(in, false);

				in.genData.addStat(place, std::move(res));
			}
			else
			{
				in.genData.addStat(place, StatementType::MOD_DEF<In>{ modName, exported });
			}

			return true;
		}
		return false;
	}
}