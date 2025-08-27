module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <ranges>
#include <string>

export module hack;
import slu.num;
import slu.ast.make;

namespace slu::hack
{
	export template<bool isFields>
	auto hackFunc(auto& itm, auto place, auto& mpDb,auto& patStack,auto& exprStack) {
		if constexpr (isFields)
		{
			for (auto& i : std::views::reverse(itm.items))
				patStack.push_back(&i.pat);
			for (auto& i : itm.items)
				exprStack.emplace_back(parse::mkFieldIdx<true>(place, itm.name, i.name));
		}
		else
		{
			for (auto& i : std::views::reverse(itm.items))
				patStack.push_back(&i);
			for (size_t i = 0; i < itm.items.size(); i++)
			{
				lang::PoolString index = mpDb.poolStr("0x" + slu::u64ToStr(i));
				exprStack.emplace_back(parse::mkFieldIdx<true>(place, itm.name, index));
			}
		}
	}
}