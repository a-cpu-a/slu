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

namespace slu::parse
{
	template<AnyInput In>
	inline bool readModStat(In& in, const Position place, const ExportData exported)
	{
		if (checkReadTextToken(in, "mod"))
		{
			std::string name = readName(in);
			const MpItmId<In> modName = in.genData.addLocalObj(name);

			if (checkReadTextToken(in, "as"))
			{
				StatementType::ModAs<In> res{};
				res.exported = exported;
				res.name = modName;

				requireToken(in, "{");
				in.genData.pushScope(in.getLoc(), std::move(name));
				res.bl = readBlockNoStartCheck<false>(in, false,false);

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