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

namespace sluaParse
{
	template<AnyInput In>
	inline bool readModStat(In& in, StatementData<In>& outData, const ExportData exported)
	{
		if (checkReadTextToken(in, "mod"))
		{
			std::string modName =
				exported
				? readName(in)
				: readName<true>(in);
			if (!exported)
			{
				if (modName == "self")
				{
					outData = StatementType::MOD_SELF{};
					return true;
				}
				if (modName == "crate")
				{
					outData = StatementType::MOD_CRATE{};
					return true;
				}
			}

			if (checkReadTextToken(in, "as"))
			{
				StatementType::MOD_DEF_INLINE<In> res{};
				res.exported = exported;
				res.name = std::move(modName);

				requireToken(in, "{");
				res.bl = readBlockNoStartCheck<false>(in, false);

				outData = std::move(res);
			}
			else
			{
				outData = StatementType::MOD_DEF{ modName,exported };
			}

			return true;
		}
		return false;
	}
}