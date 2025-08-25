/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/parse/State.hpp>

namespace slu::parse
{
	template<bool isSlu>
	inline ::slu::parse::Expr mkGlobal(::slu::ast::Position place, ::slu::lang::MpItmId name)
	{
		return { ::slu::parse::ExprType::GlobalV<isSlu>{name},
				place };
	}
	template<bool isSlu>
	inline ::slu::parse::Expr mkLocal(::slu::ast::Position place, ::slu::parse::LocalId name)
	{
		return { ::slu::parse::ExprType::Local{name},
				place };
	}
	template<bool isSlu>
	inline ::slu::parse::Expr mkNameExpr(::slu::ast::Position place, ::slu::lang::MpItmId name) {
		return mkGlobal<isSlu>(place, name);
	}
	template<bool isSlu>
	inline ::slu::parse::Expr mkNameExpr(::slu::ast::Position place, ::slu::parse::LocalId name) {
		return mkLocal<isSlu>(place, name);
	}
	template<bool isSlu>
	inline ::slu::parse::ExprDataV<isSlu> mkFieldIdx(
		::slu::ast::Position place,
		auto name,//name or local
		::slu::parse::PoolString field)
	{
		return 
			::slu::parse::ExprType::FieldV<isSlu>{
				{::slu::parse::mayBoxFrom<true>(mkNameExpr<isSlu>(place,name))},
					field};
	}
	template<bool isSlu>
	inline ::slu::parse::Expr mkFieldIdxExpr(
		::slu::ast::Position place, 
		auto name,//name or local
		::slu::parse::PoolString field)
	{
		return { mkFieldIdx(place,name,field),
				place };
	}
	template<bool isSlu,bool boxed>
	inline auto mkBoxGlobal(::slu::ast::Position place, ::slu::lang::MpItmId name) {
		return ::slu::parse::mayBoxFrom<boxed>(mkGlobal<isSlu>(place,name));
	}
	inline ::slu::parse::TableV<true> mkTbl(::slu::parse::ExprListV<true>&& exprs)
	{
		::slu::parse::TableV<true> tc;
		tc.reserve(exprs.size());
		for (auto& i : exprs)
		{
			tc.emplace_back(std::move(i));
		}
		return tc;
	}
}