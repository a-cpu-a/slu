/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/parser/State.hpp>

namespace slu::parse
{
	template<bool isSlu>
	inline std::unique_ptr<::slu::parse::LimPrefixExprV<isSlu>> mkLpe(::slu::parse::LimPrefixExprV<isSlu>&& v) {
		return std::make_unique<::slu::parse::LimPrefixExprV<isSlu>>(std::move(v));
	}
	template<bool isSlu>
	inline std::unique_ptr<::slu::parse::LimPrefixExprV<isSlu>> mkLpeVar(::slu::parse::MpItmIdV<isSlu> name)
	{
		return ::slu::parse::mkLpe<isSlu>(
			::slu::parse::LimPrefixExprType::VARv<isSlu>{
			.v = ::slu::parse::VarV<isSlu>{
				.base = ::slu::parse::BaseVarType::NAMEv<isSlu>{
					.v = name
		} } });
	}
	template<bool isSlu>
	inline std::unique_ptr<::slu::parse::LimPrefixExprV<isSlu>> mkLpeVar(::slu::parse::MpItmIdV<isSlu> name, ::slu::parse::MpItmIdV<isSlu> sub)
	{
		::slu::parse::VarV<isSlu> tmp{
			.base = ::slu::parse::BaseVarType::NAMEv<isSlu>{.v = name}
		};
		tmp.sub.emplace_back(::slu::parse::SubVarV<isSlu>{.idx = ::slu::parse::SubVarType::NAMEv<isSlu>{ .idx = sub }});
		return ::slu::parse::mkLpe<isSlu>(
			::slu::parse::LimPrefixExprType::VARv<isSlu>{
			.v = std::move(tmp) });
	}
	template<bool isSlu>
	inline std::unique_ptr<::slu::parse::LimPrefixExprV<isSlu>> mkLpeVar(::slu::parse::LocalId locId, ::slu::parse::MpItmIdV<isSlu> sub)
	{
		::slu::parse::VarV<isSlu> tmp{.base = locId};
		tmp.sub.emplace_back(::slu::parse::SubVarV<isSlu>{.idx = ::slu::parse::SubVarType::NAMEv<isSlu>{ .idx = sub }});
		return ::slu::parse::mkLpe<isSlu>(
			::slu::parse::LimPrefixExprType::VARv<isSlu>{
			.v = std::move(tmp) });
	}
	template<bool isSlu>
	inline std::unique_ptr<::slu::parse::LimPrefixExprV<isSlu>> mkLpeVar(::slu::parse::LocalId locId)
	{
		return ::slu::parse::mkLpe<isSlu>(
			::slu::parse::LimPrefixExprType::VARv<isSlu>{
			.v = ::slu::parse::VarV<isSlu>{
				.base = locId
			} });
	}
	template<bool isSlu>
	inline ::slu::parse::TableV<isSlu> mkTbl(::slu::parse::ExpListV<isSlu>&& exprs)
	{
		::slu::parse::TableV<isSlu> tc;
		tc.reserve(exprs.size());
		for (auto& i : exprs)
		{
			tc.emplace_back(::slu::parse::FieldType::EXPRv<isSlu>{
				.v = std::move(i)
			});
		}
		return tc;
	}
}