﻿/*
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
#include <slu/parser/BuildState.hpp>
#include <slu/parser/OpTraits.hpp>
#include <slu/visit/Visit.hpp>
#include <slu/midlevel/Operator.hpp>

namespace slu::mlvl
{
	template<typename T>
	struct LazyCompute
	{
		std::optional<T> value;

		T& get(auto&& create)
		{
			if (!value.has_value())
				value = create();
			return *value;
		}
	};

	using DesugarCfg = decltype(parse::sluCommon);

	struct InlineModule
	{
		lang::MpItmIdV<true> name;
		parse::StatListV<true> code;
	};

	struct DesugarVisitor : visit::EmptyVisitor<DesugarCfg>
	{
		using Cfg = visit::EmptyVisitor<DesugarCfg>::Cfg;
		static constexpr bool isSlu = Cfg::settings() & ::slu::parse::sluSyn;

		parse::BasicMpDb mpDb;
		LazyCompute<parse::MpItmId<Cfg>> unOpFuncs[(size_t)parse::UnOpType::ENUM_SIZE];
		LazyCompute<parse::MpItmId<Cfg>> postUnOpFuncs[(size_t)parse::PostUnOpType::ENUM_SIZE];
		LazyCompute<parse::MpItmId<Cfg>> binOpFuncs[(size_t)parse::BinOpType::ENUM_SIZE];

		// Note: Implicit bottom item: 'Any' or 'Slu'
		std::vector<std::string> abiStack;
		std::vector<bool> abiSafetyStack;
		std::vector<parse::StatList<Cfg>*> statListStack;
		std::vector<parse::Locals<Cfg>*> localsStack;
		std::vector<lang::ModPathId> mpStack;

		//Output!!
		std::vector<InlineModule> inlineModules;

		static parse::Expression<Cfg> wrapTypeExpr(parse::TypeExpr&& t)
		{
			parse::Expression<Cfg> res;
			res.place = t.place;
			res.data = std::move(t);
			return res;
		}

		//TODO: Implement the conversion logic here
		//TODO: basic desugaring:
		//TODO: [50%] operators
		//TODO: [_0%] auto-drop?
		//TODO: [_0%] for/while/repeat loops
		//TODO: [20%] destructuring (also apply op stuff after too)
		//TODO: destructuring but for fn args
		template<bool isLocal>
		void addCanonicVarStat(std::vector<parse::Sel<isLocal,
			parse::StatementType::CanonicGlobal,
			parse::StatementType::CanonicLocal>>& out,
			const bool isFirstVar,
			auto& localHolder,
			bool exported,
			parse::TypeExpr&& type,
			parse::LocalOrName<Cfg, isLocal> name,
			parse::Expression<Cfg>&& expr)
		{
			if constexpr (isLocal)
			{
				out.emplace_back(
					std::move(type),
					name,
					std::move(expr),
					exported);
			} else {
				out.emplace_back(
					std::move(type),
					isFirstVar ? std::move(localHolder.local2Mp) : parse::Locals<Cfg>(),
					name,
					std::move(expr),
					exported);
			}
		}
		template<bool isLocal>
		parse::LocalOrName<Cfg, isLocal> getSynVarName()
		{
			lang::ModPathId mp = mpStack.back();
			auto& mpData = mpDb.data->mps[mp.id];
			lang::LocalObjId obj = { mpData.id2Name.size() };
			parse::MpItmId<Cfg> name = {obj, mp};

			std::string synName = parse::getAnonName(obj.val);

			mpData.name2Id[synName] = obj;
			mpData.id2Name[obj.val] = std::move(synName);

			if constexpr (isLocal)
			{
				auto& localSpace = *localsStack.back();
				parse::LocalId id = { localSpace.size() };
				localSpace.push_back(name);
				return id;
			}
			else
				return name;
		}
		parse::TypeExpr destrSpec2TypeExpr(parse::Position place,parse::DestrSpec<Cfg>&& spec)
		{
			parse::TypeExpr te= ezmatch(spec)(
			varcase(parse::DestrSpecType::Prefix&) 
			{
				auto res = parse::TypeExpr{ parse::TypeExprDataType::ERR_INFERR{},place };
				if (!var.empty() && var[0].type == parse::UnOpType::MUT)
				{
					var.erase(var.begin());
					res.hasMut = true;
				}
				res.unOps = std::move(var);

				return res;
			},
			varcase(parse::DestrSpecType::Spat<Cfg>&) 
			{
				return parse::TypeExpr{ parse::mkLpe<isSlu>(
					parse::LimPrefixExprType::EXPR<Cfg>{std::move(var)}
				),place };
			}
			);
			visit::visitTypeExpr(*this, te);
			return te;
		}
		template<bool isLocal,bool isFields,class T>
		void convDestrLists(parse::Position place,
			auto& out,
			std::vector<parse::PatV<true, isLocal>*>& patStack,
			std::vector<parse::ExprData<Cfg>>& exprStack,
			const bool first,
			auto& localHolder,
			parse::Expression<Cfg>&& expr,
			bool exported,
			T& itm) requires(parse::AnyCompoundDestr<isLocal,T>)
		{
			parse::LocalOrNameV<isSlu, isLocal> name;
			if (itm.name.empty())
			{
				name = getSynVarName<isLocal>();
				exported = false;
			}
			else
				name = itm.name;
			addCanonicVarStat<isLocal>(out, first, localHolder,
				exported,
				destrSpec2TypeExpr(place, std::move(itm.spec)),
				name,
				std::move(expr));
			if constexpr (isFields)
			{
				for (auto& i : std::views::reverse(itm.items))
					patStack.push_back(&i.pat);
				for (auto& i : itm.items)
					exprStack.emplace_back(parse::mkLpeVar(name, i.name));
			}
			else
			{
				for (auto& i : std::views::reverse(itm.items))
					patStack.push_back(&i);
				for (size_t i = 0; i < itm.items.size(); i++)
				{
					parse::MpItmId<Cfg> index = mpDb.resolveUnknown("0x" + parse::u64ToStr(i));
					exprStack.emplace_back(parse::mkLpeVar(name, index));
				}
			}
		}

		template<bool isLocal,class VarT>
		void convVar(parse::Statement<Cfg>& stat,VarT& itm)
		{
			using Canonic = parse::Sel<isLocal,
				parse::StatementType::CanonicGlobal, 
				parse::StatementType::CanonicLocal>;

			std::vector<Canonic> out;
			std::vector<parse::PatV<true, isLocal>*> patStack;
			std::vector<parse::ExprData<Cfg>> exprStack;
			patStack.push_back(&itm.names);
			if (itm.exprs.size() == 1)
				exprStack.emplace_back(std::move(itm.exprs[0].data));
			else
			{
				exprStack.emplace_back(parse::ExprType::TABLE_CONSTRUCTOR<Cfg>{
					.v = parse::mkTbl(std::move(itm.exprs))
				});
			}

			bool first = true;
			do {
				parse::Expression<Cfg> expr;
				expr.place = stat.place;
				expr.data = std::move(exprStack.back());
				exprStack.pop_back();

				bool exported = itm.exported;//Synthetic ones are not exported anyway

				auto& pat = *patStack.back();
				ezmatch(pat)(
				varcase(const auto&) {
					throw std::runtime_error("Invalid destructuring pattern type, idx(" + std::to_string(pat.index()) + ") (basic desugar)");
				},
				varcase(const parse::PatType::DestrAny) {
					addCanonicVarStat<isLocal>(out, first, itm,
						false, 
						parse::TypeExpr{ parse::TypeExprDataType::ERR_INFERR{},stat.place },
						getSynVarName<isLocal>(),
						std::move(expr));
				},
				varcase(parse::PatType::DestrName<Cfg, isLocal>&) {
					addCanonicVarStat<isLocal>(out, first, itm,
						exported,
						destrSpec2TypeExpr(stat.place,std::move(var.spec)),
						var.name,
						std::move(expr));
				},
				varcase(parse::PatType::DestrFields<Cfg, isLocal>&) {
					convDestrLists<isLocal, true>(stat.place, out, patStack, exprStack, first, itm, std::move(expr), exported, var);
				},
				varcase(parse::PatType::DestrList<Cfg, isLocal>&) {
					convDestrLists<isLocal, false>(stat.place, out, patStack, exprStack, first, itm, std::move(expr), exported, var);
				}
				);
				first = false;
				patStack.pop_back();//consume one
			} while (!patStack.empty());

			stat.data = std::move(out[0]);

			if (out.size() == 1)
				return;

			// Insert the rest of the statements
			auto& statList = *statListStack.back();
			statList.insert(statList.end(), std::make_move_iterator(std::next(out.begin() + 1)), std::make_move_iterator(out.end()));
		}

		bool preFunctionInfo(parse::FunctionInfo<Cfg>& itm) 
		{
			if(itm.abi.empty())
				itm.abi = abiStack.empty() ? "Any" : abiStack.back();
			localsStack.push_back(&itm.local2Mp);
			return false;
		}
		void postFunctionInfo(parse::FunctionInfo<Cfg>& itm) {
			localsStack.pop_back();
		}
		bool preStatList(parse::StatList<Cfg>& itm) 
		{
			statListStack.push_back(&itm);
			return false;
		}
		void postStatList(parse::StatList<Cfg>& itm) {
			statListStack.pop_back();
		}
		bool preBlock(parse::Block<Cfg>& itm) 
		{
			mpStack.push_back(itm.mp);
			return false;
		}
		void postBlock(parse::Block<Cfg>& itm) {
			mpStack.pop_back();
		}
		bool preFile(parse::ParsedFile<Cfg>& itm)
		{
			mpStack.push_back(itm.mp);
			return false;
		}
		void postFile(parse::ParsedFile<Cfg>& itm) {
			mpStack.pop_back();
		}
		bool preExternBlock(parse::StatementType::ExternBlock<Cfg>& itm) 
		{
			abiStack.push_back(std::move(itm.abi));
			abiSafetyStack.push_back(itm.safety==parse::OptSafety::SAFE);

			visit::visitStatList(*this,itm.stats);

			abiStack.pop_back();
			abiSafetyStack.pop_back();
			return true;//Already visited the statements inside it
		}
		void postStat(parse::Statement<Cfg>& itm) 
		{
			if (std::holds_alternative<parse::StatementType::ExternBlock<Cfg>>(itm.data))
			{
				// Unwrap the extern block
				auto& block = std::get<parse::StatementType::ExternBlock<Cfg>>(itm.data);
				if (block.stats.empty())
				{
					itm.data = parse::StatementType::SEMICOLON{};
					return;
				}

				parse::StatList<Cfg> stats = std::move(block.stats);
				itm.place = stats.front().place;
				auto statData = std::move(stats.front().data);
				itm.data = std::move(statData);
				if (stats.size() == 1)
					return;
				// Insert the rest of the statements
				auto& statList = *statListStack.back();
				statList.insert(statList.end(), std::make_move_iterator(std::next(stats.begin()+1)), std::make_move_iterator(stats.end()));
			}
			else if(std::holds_alternative<parse::StatementType::MOD_DEF_INLINE<Cfg>>(itm.data))
			{
				// Unwrap the inline module
				//TODO: modules shouldnt use Block!
				auto& module = std::get<parse::StatementType::MOD_DEF_INLINE<Cfg>>(itm.data);
				inlineModules.push_back(InlineModule{ module.name, std::move(module.bl.statList) });
				itm.data = parse::StatementType::MOD_DEF<Cfg>{module.name,module.exported};
			}
			else if (std::holds_alternative<parse::StatementType::LOCAL_ASSIGN<Cfg>>(itm.data))
				convVar<true>(itm, std::get<parse::StatementType::LOCAL_ASSIGN<Cfg>>(itm.data));
			else if (std::holds_alternative<parse::StatementType::LET<Cfg>>(itm.data))
				convVar<true>(itm, std::get<parse::StatementType::LET<Cfg>>(itm.data));
			else if (std::holds_alternative<parse::StatementType::CONST<Cfg>>(itm.data))
				convVar<false>(itm, std::get<parse::StatementType::CONST<Cfg>>(itm.data));
		}

		template<bool forType,class MultiOp,class ExprT>
		bool desugarExpr(ExprT& itm)
		{
			if (std::holds_alternative<MultiOp>(itm.data))
			{
				_ASSERT(itm.unOps.empty());
				_ASSERT(itm.postUnOps.empty());
				MultiOp& ops = std::get<MultiOp>(itm.data);
				auto order = multiOpOrder<false>(ops);

				std::vector<ExprT> expStack;
				//newExpr.data = ;
				for (auto& i : order)
				{
					switch (i.kind)
					{
					case OpKind::Expr:
					{
						ExprT& parent = i.index == 0 ? *ops.first : ops.extra[i.index - 1].second;

						ExprT& newExpr = expStack.emplace_back();
						newExpr.data = std::move(parent.data);
						newExpr.place = parent.place;
						break;
					}
					case OpKind::BinOp:
					{
						//create a new expression for the binary operator

						//TODO: special handling for 'and', 'or', '~~' (maybe turn it special at the ast level?).

						auto expr2 = std::move(expStack.back());
						expStack.pop_back();
						auto& expr1 = expStack.back();
						parse::Position place = expr1.place;

						parse::ExprType::FUNC_CALL<Cfg> call;
						parse::ArgsType::EXPLIST<Cfg> list;
						if constexpr (forType)
						{
							list.v.emplace_back(wrapTypeExpr(std::move(expr1)));
							list.v.emplace_back(wrapTypeExpr(std::move(expr2)));
						} else {
							list.v.emplace_back(std::move(expr1));
							list.v.emplace_back(std::move(expr2));
						}
						call.argChain.emplace_back(parse::MpItmId<Cfg>::newEmpty(), std::move(list));

						parse::BinOpType op = ops.extra[i.index - 1].first;
						const size_t traitIdx = (size_t)op - 1; //-1 for none

						call.val = parse::mkLpeVar(binOpFuncs[traitIdx].get([&] {
							lang::ModPath name;
							name.reserve(4);
							name.emplace_back("std");
							bool isOrd = op == parse::BinOpType::LESS_THAN
								|| op == parse::BinOpType::LESS_EQUAL
								|| op == parse::BinOpType::GREATER_THAN
								|| op == parse::BinOpType::GREATER_EQUAL;
							bool isEq = op == parse::BinOpType::EQUAL
								|| op == parse::BinOpType::NOT_EQUAL;
							if (isOrd || isEq)
							{
								name.emplace_back("cmp");
								if (isOrd)
									name.emplace_back("PartialOrd");
								else
									name.emplace_back("PartialEq");
							}
							else
							{
								name.emplace_back("ops");
								if (op == parse::BinOpType::RANGE_BETWEEN)
									name.emplace_back("Boundable");//TODO: choose the name!
								else
									name.emplace_back(parse::binOpTraitNames[traitIdx]);
							}
							name.emplace_back(parse::binOpNames[traitIdx]);
							return mpDb.getItm(name);
							})
						);


						//Turn the first (moved)expr in a function call expression
						expr1.data = std::move(call);
						expr1.place = place;
						break;
					}
					case OpKind::UnOp:
					case OpKind::PostUnOp:
					{
						//TODO: special handling for '.*', '?'. -> defer to after type checking / inference

						auto& opSrcExpr = expStack[i.index];

						auto& expr = expStack.back();
						parse::Position place = expr.place;
						//Wrap
						parse::MpItmId<Cfg> name;
						parse::Lifetime* lifetime = nullptr;
						{
							if (i.kind == OpKind::UnOp)
							{
								parse::UnOpItem& op = opSrcExpr.unOps[i.opIdx];
								const size_t traitIdx = (size_t)op.type - 1; //-1 for none

								name = unOpFuncs[traitIdx].get([&] {
									lang::ModPath name;
									name.reserve(4);
									name.emplace_back("std");
									name.emplace_back("ops");
									if (op.type == parse::UnOpType::RANGE_BEFORE)
										name.emplace_back("Boundable");//TODO: choose the name!
									else
										name.emplace_back(parse::unOpTraitNames[traitIdx]);
									name.emplace_back(parse::unOpNames[traitIdx]);
									return mpDb.getItm(name);
									});
								if (!op.life.empty())
								{
									_ASSERT(op.type == parse::UnOpType::TO_REF
										|| op.type == parse::UnOpType::TO_REF_MUT
										|| op.type == parse::UnOpType::TO_REF_CONST
										|| op.type == parse::UnOpType::TO_REF_SHARE);
									lifetime = &op.life;
								}
								//Else: its inferred, or doesnt exist
							}
							else if (i.kind == OpKind::PostUnOp)
							{
								parse::PostUnOpType op = opSrcExpr.postUnOps.at(i.opIdx);
								const size_t traitIdx = (size_t)op - 1; //-1 for none

								name = postUnOpFuncs[traitIdx].get([&] {
									//TODO: implement post-unop func name selection
									if (op == parse::PostUnOpType::RANGE_AFTER)
									{
										lang::ModPath name;
										name.reserve(4);
										name.emplace_back("std");
										name.emplace_back("ops");
										name.emplace_back("Boundable");//TODO: choose the name!
										name.emplace_back(parse::postUnOpNames[traitIdx]);
										return mpDb.getItm(name);
									}
									//TODO!!!
									return parse::MpItmId<Cfg>::newEmpty();
									});
							}
						}
						parse::ExprType::FUNC_CALL<Cfg> call;
						parse::ArgsType::EXPLIST<Cfg> list;
						if constexpr (forType)
							list.v.emplace_back(wrapTypeExpr(std::move(expr)));
						else
							list.v.emplace_back(std::move(expr));

						if (lifetime != nullptr)
						{
							parse::Expression<Cfg> lifetimeExpr;
							lifetimeExpr.place = place;
							lifetimeExpr.data = parse::ExprType::LIFETIME{ std::move(*lifetime) };
							list.v.emplace_back(std::move(lifetimeExpr));
						}
						call.argChain.emplace_back(parse::MpItmId<Cfg>::newEmpty(), std::move(list));
						call.val = parse::mkLpeVar(name);

						//Turn the (moved)expr in a function call expression
						expr.data = std::move(call);
						expr.place = place;
						break;
					}
					};
				}
				//desugar operators into trait func calls!
				return true;
			}
			return false;
		}

		bool preTypeExpr(parse::TypeExpr& itm) 
		{
			using MultiOp = parse::TypeExprDataType::MULTI_OP;
			return desugarExpr<true,MultiOp>(itm);
		}
		bool preExpr(parse::Expression<Cfg>& itm) 
		{
			using MultiOp = parse::ExprType::MULTI_OPERATION<Cfg>;
			return desugarExpr<false,MultiOp>(itm);
		}; 
	};

	inline std::vector<InlineModule> basicDesugar(parse::BasicMpDbData& mpDbData,parse::ParsedFileV<true>& itm)
	{
		DesugarVisitor vi{ {},parse::BasicMpDb{ &mpDbData } };

		visit::visitFile(vi, itm);

		return std::move(vi.inlineModules);
	}
}