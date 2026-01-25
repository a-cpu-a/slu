/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;
#include <utility>
#include <vector>
export module slu.ast.make;
import slu.ast.pos;
import slu.ast.state;
import slu.ast.state_decls;
import slu.lang.basic_state;

namespace slu::parse //TODO: ast
{
	export template<bool isSlu>
	::slu::parse::Expr mkGlobal(
	    ::slu::ast::Position place, ::slu::lang::MpItmId name)
	{
		return {::slu::parse::ExprType::GlobalV<isSlu>{name}, place};
	}
	export template<bool isSlu>
	::slu::parse::Expr mkLocal(
	    ::slu::ast::Position place, ::slu::parse::LocalId name)
	{
		return {::slu::parse::ExprType::Local{name}, place};
	}
	export template<bool isSlu>
	::slu::parse::Expr mkNameExpr(
	    ::slu::ast::Position place, ::slu::lang::MpItmId name)
	{
		return mkGlobal<isSlu>(place, name);
	}
	export template<bool isSlu>
	::slu::parse::Expr mkNameExpr(
	    ::slu::ast::Position place, ::slu::parse::LocalId name)
	{
		return mkLocal<isSlu>(place, name);
	}
	export template<bool isSlu>
	::slu::parse::ExprDataV<isSlu> mkFieldIdx(::slu::ast::Position place,
	    auto name, //name or local
	    ::slu::lang::PoolString field)
	{
		return ::slu::parse::ExprType::FieldV<isSlu>{
		    {::slu::parse::mayBoxFrom<true>(mkNameExpr<isSlu>(place, name))},
		    field};
	}
	export template<bool isSlu>
	::slu::parse::Expr mkFieldIdxExpr(::slu::ast::Position place,
	    auto name, //name or local
	    ::slu::lang::PoolString field)
	{
		return {mkFieldIdx(place, name, field), place};
	}
	export template<bool isSlu, bool boxed>
	auto mkBoxGlobal(::slu::ast::Position place, ::slu::lang::MpItmId name)
	{
		return ::slu::parse::mayBoxFrom<boxed>(mkGlobal<isSlu>(place, name));
	}
	export ::slu::parse::TableV<true> mkTbl(::slu::parse::ExprList&& exprs)
	{
		::slu::parse::TableV<true> tc;
		tc.reserve(exprs.size());
		for (auto& i : exprs)
		{
			tc.emplace_back(std::move(i));
		}
		return tc;
	}
} //namespace slu::parse