/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <thread>
#include <variant>
#include <slu/lang/BasicState.hpp>
#include <slu/parser/State.hpp>

namespace slu::mlvl
{
	inline void desugerDestrSpec(parse::DestrSpecV<true>& itm)
	{
		ezmatch(itm)(
		varcase(parse::DestrSpecType::SpatV<true>&) {
			desugarExpr(var);
		},
		varcase(parse::DestrSpecType::Type&) {
			desugarTypeExp(var);
		},
		varcase(parse::DestrSpecType::Prefix&) {}
		);
	}
	inline void desugarPat(parse::PatV<true>& itm)
	{
		ezmatch(itm)(
		varcase(const parse::PatType::DestrAny){ },

		varcase(parse::PatType::SimpleV<true>&) {
			desugarExpr(var);
		},

		varcase(parse::PatType::DestrFieldsV<true>&) {
			desugerDestrSpec(var.spec);
			for (auto& i : var.items)
				desugarPat(i.pat);
		},
		varcase(parse::PatType::DestrListV<true>&) {
			desugerDestrSpec(var.spec);
			for (auto& i : var.items)
				desugarPat(i);
		},

		varcase(parse::PatType::DestrNameV<true>&) {
			desugerDestrSpec(var.spec);
		},
		varcase(parse::PatType::DestrNameRestrictV<true>&) {
			desugerDestrSpec(var.spec);
			desugarExpr(var.restriction);
		}
		);
	}
	inline void desugarVar(parse::VarV<true>& itm)
	{
		ezmatch(itm.base)(
		varcase(parse::BaseVarType::EXPRv<true>&) { desugarExpr(var.start); },
		varcase(parse::BaseVarType::NAMEv<true>&){},
		varcase(const parse::BaseVarType::Root){}
		);
		for (auto& i : itm.sub)
		{
			desugarArgChain(i.funcCalls);
			ezmatch(i.idx)(
			varcase(parse::SubVarType::NAMEv<true>&){},
			varcase(parse::SubVarType::EXPRv<true>&){desugarExpr(var.idx); },
			varcase(parse::SubVarType::DEREF){}
			);
		}
	}
	inline void desugarVarList(std::span<parse::VarV<true>> itm)
	{
		for (auto& i : itm)
			desugarVar(i);
	}
	inline void desugarTypeExp(parse::TypeExpr& itm)
	{
		//TODO
	}
	inline void desugarExpr(parse::ExpressionV<true>& itm)
	{
		//TODO
	}
	inline void desugarExpList(parse::ExpListV<true>& itm)
	{
		for (auto& i : itm)
			desugarExpr(i);
	}
	inline void desugarLimPrefixExpr(parse::LimPrefixExprV<true>& itm)
	{
		ezmatch(itm)(
		varcase(parse::LimPrefixExprType::EXPRv<true>&) { desugarExpr(var.v); },
		varcase(parse::LimPrefixExprType::VARv<true>&) { desugarVar(var.v); }
		);
	}
	inline void desugarArgChain(std::span<parse::ArgFuncCallV<true>> itm)
	{
		for (auto& i : itm)
		{
			ezmatch(i.args)(
			varcase(parse::ArgsType::EXPLISTv<true>&){ desugarExpList(var.v); },
			varcase(parse::ArgsType::TABLEv<true>&){ desugarTable(var.v); },
			varcase(const parse::ArgsType::LITERAL&){}
			);
		}
	}
	inline void desugarTable(parse::TableConstructorV<true>& itm)
	{
		for (auto& i : itm)
		{
			ezmatch(i)(
				//TODO Field
			);
		}
	}
	inline void desugarParams(parse::ParamListV<true>& itm)
	{
		for (auto& i : itm)
			desugarPat(i.name);
	}
	template<typename T>
	concept AnyUndesugarableStatement =
		std::same_as<T, parse::StatementType::GOTOv<true>>
		|| std::same_as<T, parse::StatementType::LABELv<true>>
		|| std::same_as<T, parse::StatementType::SEMICOLON>
		|| std::same_as<T, parse::StatementType::BREAK>
		|| std::same_as<T, parse::StatementType::USE>
		|| std::same_as<T, parse::StatementType::MOD_DEFv<true>>;
	inline void desugarStat(parse::StatementV<true>& itm)
	{
		ezmatch(itm.data)(
		varcase(parse::StatementType::ASSIGNv<true>&) {
			desugarExpList(var.exprs);
			desugarVarList(var.vars);
		},
		varcase(parse::StatementType::LOCAL_ASSIGNv<true>&) {
			desugarExpList(var.exprs);
			desugarPat(var.names);
		},
		varcase(parse::StatementType::LETv<true>&) {
			desugarExpList(var.exprs);
			desugarPat(var.names);
		},
		varcase(parse::StatementType::CONSTv<true>&) {
			desugarExpList(var.exprs);
			desugarPat(var.names);
		},
		varcase(parse::StatementType::FUNC_CALLv<true>&) {
			desugarLimPrefixExpr(*var.val);
			desugarArgChain(var.argChain);
		},
		varcase(parse::StatementType::BLOCKv<true>&) {
			desugarBlock(var.bl);
		},
		varcase(parse::StatementType::IfCondV<true>&) {
			desugarSoe(*var.bl);
			if(var.elseBlock.has_value())
				desugarSoe(**var.elseBlock);
			desugarExpr(*var.cond);
			for (auto& [cond,soe] : var.elseIfs)
			{
				desugarExpr(cond);
				desugarSoe(soe);
			}
		},
		varcase(parse::StatementType::WHILE_LOOPv<true>&) {
			desugarBlock(var.bl);
			desugarExpr(var.cond);
		},
		varcase(parse::StatementType::REPEAT_UNTILv<true>&) {
			desugarBlock(var.bl);
			desugarExpr(var.cond);
		},
		varcase(parse::StatementType::FOR_LOOP_NUMERICv<true>&) {
			desugarBlock(var.bl);
			desugarExpr(var.start);
			desugarExpr(var.end);
			if (var.step.has_value())
				desugarExpr(*var.step);
			desugarPat(var.varName);
		},
		varcase(parse::StatementType::FOR_LOOP_GENERICv<true>&) {
			desugarBlock(var.bl);
			desugarExpr(var.exprs);
			desugarPat(var.varNames);
		},
		varcase(parse::StatementType::StructV<true>&) {
			desugarParams(var.params);
			desugarTypeExp(var.type);
		},
		varcase(parse::StatementType::UnionV<true>&) {
			desugarParams(var.params);
			desugarTable(var.type);
		},
		varcase(parse::StatementType::ExternBlockV<true>&) {
			//TODO
		},
		varcase(parse::StatementType::UnsafeBlockV<true>&) {
			//TODO
		},
		varcase(const parse::StatementType::UNSAFE_LABEL) {
			//TODO
		},
		varcase(const parse::StatementType::SAFE_LABEL) {
			//TODO
		},
		varcase(parse::StatementType::DROPv<true>&) {
			desugarExpr(var.expr);
		},
		varcase(parse::StatementType::MOD_DEF_INLINEv<true>&) {
			desugarBlock(var.bl);
		},
			//Ignore these
		varcase(const AnyUndesugarableStatement auto&) {}
		);
	}
	inline void desugarSoe(parse::SoeV<true>& itm)
	{
		ezmatch(itm)(
		varcase(parse::BlockV<true>&) {
			desugarBlock(var);
		},
		varcase(parse::ExpressionV<true>&) {
			desugarExpr(var);
		}
		);
	}
	inline void desugarBlock(parse::BlockV<true>& itm)
	{
		for (auto& i : itm.statList)
			desugarStat(i);
		//if (itm.hadReturn)
		desugarExpList(itm.retExprs);//Will be empty if no return is present
	}
	inline void basicDesugar(parse::ParsedFileV<true>& itm)
	{
		//TODO: Implement the conversion logic here
		//TODO: basic desugaring:
		//TODO: operators
		//TODO: auto-drop?
		//TODO: for/while/repeat loops
		desugarBlock(itm.code);
	}
}