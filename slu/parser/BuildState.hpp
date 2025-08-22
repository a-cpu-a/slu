/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/parser/State.hpp>

namespace slu::parse
{
	template<bool isSlu>
	inline ::slu::parse::ExprV<isSlu> mkGlobal(::slu::parse::Position place, ::slu::lang::MpItmId name)
	{
		return {::slu::parse::BaseExprV<isSlu>{
			::slu::parse::ExprType::GlobalV<isSlu>{name},
				place
		} };
	}
	template<bool isSlu>
	inline ::slu::parse::ExprV<isSlu> mkLocal(::slu::parse::Position place, ::slu::parse::LocalId name)
	{
		return {::slu::parse::BaseExprV<isSlu>{
			::slu::parse::ExprType::Local{name},
				place
		} };
	}
	template<bool isSlu>
	inline ::slu::parse::ExprV<isSlu> mkNameExpr(::slu::parse::Position place, ::slu::lang::MpItmId name) {
		return mkGlobal<isSlu>(place, name);
	}
	template<bool isSlu>
	inline ::slu::parse::ExprV<isSlu> mkNameExpr(::slu::parse::Position place, ::slu::parse::LocalId name) {
		return mkLocal<isSlu>(place, name);
	}
	template<bool isSlu>
	inline ::slu::parse::ExprDataV<isSlu> mkFieldIdx(
		::slu::parse::Position place,
		auto name,//name or local
		::slu::parse::PoolString field)
	{
		return 
			::slu::parse::ExprType::FieldV<isSlu>{
				{::slu::parse::mayBoxFrom<true>(mkNameExpr<isSlu>(place,name))},
					field};
	}
	template<bool isSlu>
	inline ::slu::parse::ExprV<isSlu> mkFieldIdxExpr(
		::slu::parse::Position place, 
		auto name,//name or local
		::slu::parse::PoolString field)
	{
		return {::slu::parse::BaseExprV<isSlu>{
				mkFieldIdx(place,name,field),
				place
		} };
	}
	template<bool isSlu,bool boxed>
	inline auto mkBoxGlobal(::slu::parse::Position place, ::slu::lang::MpItmId name) {
		return ::slu::parse::mayBoxFrom<boxed>(mkGlobal<isSlu>(place,name));
	}
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