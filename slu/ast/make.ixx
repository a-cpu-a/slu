module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <utility>
export module slu.ast.make;
import slu.ast.pos;
import slu.ast.state;
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