/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/RequireToken.hpp>
#include <slu/parser/adv/ReadName.hpp>

namespace slua::parse
{
	template<AnyInput In>
	inline bool readModStat(In& in, const Position place, const ExportData exported)
	{
		if (checkReadTextToken(in, "mod"))
		{
			const MpItmId<In> modName = in.genData.resolveUnknown(readName(in));

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