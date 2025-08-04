/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/parser/State.hpp>

namespace slu::parse
{
	template<bool isSlu>
	inline ::slu::parse::TableV<isSlu> mkTbl(::slu::parse::ExprListV<isSlu>&& exprs)
	{
		::slu::parse::TableV<isSlu> tc;
		tc.reserve(exprs.size());
		for (auto& i : exprs)
		{
			tc.emplace_back(std::move(i));
		}
		return tc;
	}
}