module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <memory>
#include <ranges>
#include <stdexcept>

#include <slu/Panic.hpp>
#include <slu/ext/CppMatch.hpp>
export module slu.mlvl.basic_desugar;

import slu.num;
import slu.settings;
import slu.ast.enums;
import slu.ast.make;
import slu.ast.mp_data;
import slu.ast.op_info;
import slu.ast.op_order;
import slu.ast.pos;
import slu.ast.small_enum_list;
import slu.ast.state;
import slu.ast.state_decls;
import slu.ast.type;
import slu.lang.basic_state;
import slu.mlvl.resolve_type;
import slu.visit.empty;
import slu.visit.visit;

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

	export struct InlineModule
	{
		lang::MpItmId name;
		parse::StatListV<true> code;
	};

	struct DesugarVisitor : visit::EmptyVisitor<DesugarCfg>
	{
		using Cfg = DesugarCfg;
		static constexpr bool isSlu = true;

		parse::BasicMpDb mpDb;
		LazyCompute<lang::MpItmId> unOpFuncs[(size_t)ast::UnOpType::ENUM_SIZE];
		LazyCompute<lang::MpItmId> postUnOpFuncs[(size_t)ast::PostUnOpType::ENUM_SIZE];
		LazyCompute<lang::MpItmId> binOpFuncs[(size_t)ast::BinOpType::ENUM_SIZE];

		// Note: Implicit bottom item: 'Any' or 'Slu'
		std::vector<std::string> abiStack;
		std::vector<bool> abiSafetyStack;
		std::vector<parse::StatList<Cfg>*> statListStack;
		std::vector<parse::Locals<Cfg>*> localsStack;
		std::vector<lang::ModPathId> mpStack;
		std::vector<parse::BasicModPathData*> mpDataStack;

		//Output!!
		std::vector<InlineModule> inlineModules;


		//TODO: [50%] operators
		//TODO: [_0%] auto-drop?
		//TODO: [_0%] for/while/repeat loops? maybe unnececary?
		//TODO: destructuring for fn args
		//TODO: consider adding a splice statement, instead of inserting statements (OR FIX THE UB FOR THAT!)


		void postUse(parse::StatType::Use& itm)
		{
			ezmatch(itm.useVariant)(
			varcase(parse::UseVariantType::EVERYTHING_INSIDE&) {
				//TODO
			},
			varcase(parse::UseVariantType::AS_NAME&) {
				auto& localMp = mpDb.data->mps[var.mp.id];
				localMp.addItm(var.id, parse::ItmType::Alias{ itm.base });
			},
			varcase(parse::UseVariantType::IMPORT&) {
				//var.name -> local
				auto& localMp = mpDb.data->mps[var.name.mp.id];
				localMp.addItm(var.name.id, parse::ItmType::Alias{itm.base});
			},
			varcase(parse::UseVariantType::LIST_OF_STUFF&) {
				auto& localMp = *mpDataStack.back();
				auto p = mpDb.data->mp2Id.find(itm.base.asVmp(mpDb));
				if (p == mpDb.data->mp2Id.end())
				{
					//TODO error
					return;
				}
				auto& baseMp = mpDb.data->mps[p->second.id];
				for (auto& i : var)
				{
					std::string_view name =i.asSv(mpDb);
					/*
					if (name == "self")
					{
						std::string_view selfName = itm.base.asSv(mpDb);
						localMp.
						//TODO
						localMp.addItm(itm.base.id, parse::ItmType::Alias{ itm.base });
						continue;
					}*/
					lang::MpItmId nonLocal;
					nonLocal.mp = p->second;
					nonLocal.id= baseMp.get(name);
					
					localMp.addItm(i.id, parse::ItmType::Alias{ nonLocal });
				}
			}
			);
		}
		void mkFuncStatItm(lang::LocalObjId obj,std::string&& abi,std::optional<std::unique_ptr<parse::Expr>>&& ret, parse::ParamList& params,lang::ExportData exported,const bool hasCode)
		{
			auto& localMp = *mpDataStack.back();

			parse::ItmType::Fn res;
			res.abi = std::move(abi);
			res.exported = exported;
			if (ret.has_value())
				res.ret = resolveTypeExpr(mpDb, std::move(**ret));
			else
				res.ret = parse::ResolvedType{ .base = parse::RawTypeKind::Struct{},.size = 0 };

			res.args.reserve(params.size());
			if(hasCode)
				res.argLocals.reserve(params.size());
			for (auto& i : params)
			{
				res.args.emplace_back(resolveTypeExpr(mpDb, std::move(i.type)));
				if(hasCode)
				{
					ezmatch(i.name)(
					varcase(const parse::LocalId&) {
						res.argLocals.push_back(var);
					},
					varcase(const lang::MpItmId&) {}
					);
				}
			}
			params.clear();

			localMp.addItm(obj, std::move(res));
		}
		void postAnyFuncDeclStat(parse::StatType::FunctionDecl<Cfg>& itm) {
			mkFuncStatItm(itm.name.id, std::move(itm.abi), std::move(itm.retType), itm.params,itm.exported,false);
		}
		void postAnyFuncDefStat(parse::StatType::Function& itm) {
			mkFuncStatItm(itm.name.id, std::move(itm.func.abi), std::move(itm.func.retType), itm.func.params, itm.exported,true);
		}

		bool preGlobal(parse::ExprType::Global<Cfg>& itm) {
			if (mpDb.isUnknown(itm))
			{
				parse::BasicModPathData& mp = mpDb.data->mps[itm.mp.id];
				std::string_view item = mp.id2Name[itm.id.val];

				std::string_view start = item;
				if (mp.path.size() > 1)
					start = mp.path[1];

				for (const auto& i : mpStack)
				{
					parse::BasicModPathData& testMp = mpDb.data->mps[i.id];
					//Stored with +1 !!!
					size_t k = 0;
					for (std::string_view v : testMp.id2Name)
					{
						k++;
						if (v != start)
							continue;

						if (mp.path.size() > 1)
						{
							lang::ModPath tmpMp;
							tmpMp.reserve(testMp.path.size() + mp.path.size() - 1 + 1);
							tmpMp.insert(tmpMp.end(), testMp.path.begin(), testMp.path.end());
							tmpMp.insert(tmpMp.end(), mp.path.begin() + 1, mp.path.end());
							tmpMp.emplace_back(item);

							itm.mp = mpDb.get<false>(tmpMp);
							itm.id = mpDb.data->mps[itm.mp.id].get(item);
							return false;
						}
						itm.mp = i;
						itm.id.val = k-1;
						return false;
					}
				}
				//Must be a error or something from a `use ::*`
			}
			return false;
		}

		template<bool isLocal>
		void addCanonicVarStat(std::vector<parse::Sel<isLocal,
			parse::StatType::CanonicGlobal,
			parse::StatType::CanonicLocal>>& out,
			const bool isFirstVar,
			auto& localHolder,
			bool exported,
			parse::ResolvedType&& type,
			parse::LocalOrName<Cfg, isLocal> name,
			parse::Expr&& expr)
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
		//template<bool isLocal>
		//parse::LocalOrName<Cfg, isLocal> getSynVarName()
		//{
		//	lang::ModPathId mp = mpStack.back();
		//	auto& mpData = mpDb.data->mps[mp.id];
		//	lang::LocalObjId obj = { mpData.id2Name.size() };
		//	lang::MpItmId name = {obj, mp};
		//
		//	std::string synName = parse::getAnonName(obj.val);
		//
		//	mpData.name2Id[synName] = obj;
		//	mpData.id2Name[obj.val] = std::move(synName);
		//
		//	if constexpr (isLocal)
		//	{
		//		auto& localSpace = *localsStack.back();
		//		parse::LocalId id = { localSpace.names.size() };
		//		localSpace.names.push_back(name);
		//		return id;
		//	}
		//	else
		//		return name;
		//}
		parse::ResolvedType destrSpec2Type(ast::Position place,parse::DestrSpec&& spec)
		{
			parse::Expr te = ezmatch(spec)(
			varcase(parse::DestrSpecType::Prefix&) 
			{
				return parse::Expr{ parse::ExprType::Infer{},place,std::move(var) };
			},
			varcase(parse::DestrSpecType::Spat&)  {
				return std::move(var);
			}
			);
			visit::visitTypeExpr(*this, te);
			return resolveTypeExpr(mpDb,std::move(te));
		}
		template<bool isLocal,bool isFields,class T>
		void convDestrLists(ast::Position place,
			auto& out,
			std::vector<parse::PatV<true, isLocal>*>& patStack,
			std::vector<parse::ExprData<Cfg>>& exprStack,
			const bool first,
			auto& localHolder,
			parse::Expr&& expr,
			bool exported,
			T& itm) requires(parse::AnyCompoundDestr<isLocal,T>)
		{
			//TODO: do this for synthetic names: exported = false;
			addCanonicVarStat<isLocal>(out, first, localHolder,
				exported,
				destrSpec2Type(place, std::move(itm.spec)),
				itm.name,
				std::move(expr));
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

		template<bool isLocal,class VarT>
		void convVar(parse::Stat& stat,VarT& itm)
		{
			using Canonic = parse::Sel<isLocal,
				parse::StatType::CanonicGlobal, 
				parse::StatType::CanonicLocal>;

			std::vector<Canonic> out;
			std::vector<parse::PatV<true, isLocal>*> patStack;
			std::vector<parse::ExprData<Cfg>> exprStack;
			patStack.push_back(&itm.names);
			if (itm.exprs.size() == 1)
				exprStack.emplace_back(std::move(itm.exprs[0].data));
			else
				exprStack.emplace_back(parse::mkTbl(std::move(itm.exprs)));

			bool first = true;
			do {
				parse::Expr expr;
				expr.place = stat.place;
				expr.data = std::move(exprStack.back());
				exprStack.pop_back();

				bool exported = itm.exported;//Synthetic ones are not exported anyway

				auto& pat = *patStack.back();
				ezmatch(pat)(
				varcase(const auto&) {
					throw std::runtime_error("Invalid destructuring pattern type, idx(" + std::to_string(pat.index()) + ") (basic desugar)");
				},
				varcase(const parse::PatType::DestrAny<Cfg, isLocal>) {
					addCanonicVarStat<isLocal>(out, first, itm,
						false, 
						parse::ResolvedType::getInferred(),
						var,
						std::move(expr));
				},
				varcase(parse::PatType::DestrName<Cfg, isLocal>&) {
					addCanonicVarStat<isLocal>(out, first, itm,
						exported,
						destrSpec2Type(stat.place,std::move(var.spec)),
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

		bool preTrait(parse::StatType::Trait& itm) 
		{
			//TODO
			return false;
		}

		bool preImpl(parse::StatType::Impl& itm)
		{
			//TODO
			return false;
		}

		bool preFunctionInfo(parse::FunctionInfo& itm) 
		{
			if(itm.abi.empty())
				itm.abi = abiStack.empty() ? "Any" : abiStack.back();
			return false;
		}
		bool preLocals(parse::Locals<Cfg>& itm)
		{
			localsStack.push_back(&itm);
			return false;
		}
		void postLocals(parse::Locals<Cfg>& itm) {
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
			mpDataStack.push_back(&mpDb.data->mps[itm.mp.id]);
			return false;
		}
		void postBlock(parse::Block<Cfg>& itm) {
			mpStack.pop_back();
			mpDataStack.pop_back();
		}
		bool preFile(parse::ParsedFile& itm)
		{
			mpStack.push_back(itm.mp);
			mpDataStack.push_back(&mpDb.data->mps[itm.mp.id]);
			return false;
		}
		void postFile(parse::ParsedFile& itm) {
			mpStack.pop_back();
			mpDataStack.pop_back();
		}
		bool preExternBlock(parse::StatType::ExternBlock<Cfg>& itm) 
		{
			abiStack.push_back(std::move(itm.abi));
			abiSafetyStack.push_back(itm.safety==ast::OptSafety::SAFE);

			visit::visitStatList(*this,itm.stats);

			abiStack.pop_back();
			abiSafetyStack.pop_back();
			return true;//Already visited the statements inside it
		}
		void postStat(parse::Stat& itm) 
		{
			/*
			if (std::holds_alternative<parse::StatType::ExternBlock<Cfg>>(itm.data))
			{
				// Unwrap the extern block
				auto& block = std::get<parse::StatType::ExternBlock<Cfg>>(itm.data);
				if (block.stats.empty())
				{
					itm.data = parse::StatType::Semicol{};
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
			}*/
			if(std::holds_alternative<parse::StatType::ModAs<Cfg>>(itm.data))
			{// Unwrap the inline module
				auto& module = std::get<parse::StatType::ModAs<Cfg>>(itm.data);
				inlineModules.push_back(InlineModule{ module.name, std::move(module.code) });
				itm.data = parse::StatType::Mod<Cfg>{module.name,module.exported};
			}
			else if (std::holds_alternative<parse::StatType::Local<Cfg>>(itm.data))
				convVar<true>(itm, std::get<parse::StatType::Local<Cfg>>(itm.data));
			else if (std::holds_alternative<parse::StatType::Let<Cfg>>(itm.data))
				convVar<true>(itm, std::get<parse::StatType::Let<Cfg>>(itm.data));
			else if (std::holds_alternative<parse::StatType::Const<Cfg>>(itm.data))
				convVar<false>(itm, std::get<parse::StatType::Const<Cfg>>(itm.data));
		}

		void desugarUnOp(parse::Expr& expr,
			std::vector<parse::UnOpItem>& unOps, 
			ast::SmallEnumList<ast::PostUnOpType>& postUnOps,
			size_t opIdx,
			bool isSufOp)
		{
			ast::Position place = expr.place;
			//Wrap
			lang::MpItmId name;
			parse::Lifetime* lifetime = nullptr;

			if (isSufOp)
			{
				ast::PostUnOpType op = postUnOps.at(opIdx);
				const size_t traitIdx = (size_t)op - 1; //-1 for none

				name = postUnOpFuncs[traitIdx].get([&] {
					//TODO: implement post-unop func name selection
					if (op != ast::PostUnOpType::TRY)
					{
						lang::ModPath name;
						name.reserve(4);
						name.emplace_back("std");
						name.emplace_back("ops");
						name.emplace_back(ast::postUnOpTraitNames[traitIdx]);
						name.emplace_back(ast::postUnOpNames[traitIdx]);
						return mpDb.getItm(name);
					}
					//TODO: special handling for '?'. -> defer to after type checking / inference?
					return lang::MpItmId::newEmpty();
					});
			}
			else
			{
				parse::UnOpItem& op = unOps[opIdx];
				const size_t traitIdx = (size_t)op.type - 1; //-1 for none

				name = unOpFuncs[traitIdx].get([&] {
					lang::ModPath name;
					name.reserve(4);
					name.emplace_back("std");
					name.emplace_back("ops");
					name.emplace_back(ast::unOpTraitNames[traitIdx]);
					name.emplace_back(ast::unOpNames[traitIdx]);
					return mpDb.getItm(name);
					});
				if (!op.life.empty())
				{
					Slu_assert(op.type == ast::UnOpType::REF
						|| op.type == ast::UnOpType::REF_MUT
						|| op.type == ast::UnOpType::REF_CONST
						|| op.type == ast::UnOpType::REF_SHARE);
					lifetime = &op.life;
				}
				//Else: its inferred, or doesnt exist
			}
			
			parse::ExprType::SelfCall call;
			parse::ArgsType::ExprList list;

			if (lifetime != nullptr)
			{
				parse::Expr lifetimeExpr;
				lifetimeExpr.place = place;
				lifetimeExpr.data = parse::ExprType::Lifetime{ std::move(*lifetime) };
				list.emplace_back(std::move(lifetimeExpr));
			}
			call.args = std::move(list);
			call.v = parse::mayBoxFrom<true>(std::move(expr));
			call.method = name;

			//Turn the (moved)expr in a function call expression
			expr.data = std::move(call);
			expr.place = place;
		}

		using ExprT = parse::Expr;
		bool preExpr(ExprT& itm)
		{
			using MultiOp = parse::ExprType::MultiOp<Cfg>;

			if (std::holds_alternative<MultiOp>(itm.data))
			{
				Slu_assert(itm.unOps.empty());
				Slu_assert(itm.postUnOps.empty());
				MultiOp& ops = std::get<MultiOp>(itm.data);
				auto order = ast::multiOpOrder(ops);

				std::vector<ExprT> expStack;
				for (auto& i : order)
				{
					switch (i.kind)
					{
					case ast::OpKind::Expr:
					{
						ExprT& parent = i.index == 0 ? *ops.first : ops.extra[i.index - 1].second;

						ExprT& newExpr = expStack.emplace_back();
						newExpr.data = std::move(parent.data);
						newExpr.place = parent.place;
						visit::visitExpr(*this, newExpr);
						break;
					}
					case ast::OpKind::BinOp:
					{
						//create a new expression for the binary operator

						//TODO: special handling for 'and', 'or', '~~' (maybe turn it special at the ast level?).

						auto expr2 = std::move(expStack.back());
						expStack.pop_back();
						auto& expr1 = expStack.back();
						ast::Position place = expr1.place;

						parse::ExprType::SelfCall call;
						parse::ArgsType::ExprList list;
						call.v = parse::mayBoxFrom<true>(std::move(expr1));
						list.emplace_back(std::move(expr2));
						call.args = std::move(list);

						ast::BinOpType op = ops.extra[i.index - 1].first;
						const size_t traitIdx = (size_t)op - 1; //-1 for none

						call.method = binOpFuncs[traitIdx].get([&] {
							lang::ModPath name;
							name.reserve(4);
							name.emplace_back("std");
							bool isOrd = op == ast::BinOpType::LT
								|| op == ast::BinOpType::LE
								|| op == ast::BinOpType::GT
								|| op == ast::BinOpType::GE;
							bool isEq = op == ast::BinOpType::EQ
								|| op == ast::BinOpType::NE;
							if (isOrd || isEq)
								name.emplace_back("cmp");
							else
								name.emplace_back("ops");
							name.emplace_back(ast::binOpTraitNames[traitIdx]);
							name.emplace_back(ast::binOpNames[traitIdx]);
							return mpDb.getItm(name);
						});


						//Turn the first (moved)expr in a function call expression
						expr1.data = std::move(call);
						expr1.place = place;
						break;
					}
					case ast::OpKind::UnOp:
					case ast::OpKind::PostUnOp:
					{
						auto& opSrcExpr = expStack[i.index];
						desugarUnOp(expStack.back(), opSrcExpr.unOps, opSrcExpr.postUnOps,i.opIdx, i.kind== ast::OpKind::PostUnOp);
						break;
					}
					};
				}
				//dont call normal stuff!
				return true;
			}
			else
			{//Desugar un/post ops alone.
				const std::vector<bool> order = ast::unaryOpOrder(itm);

				std::vector<parse::UnOpItem> preOps = std::move(itm.unOps);
				ast::SmallEnumList<ast::PostUnOpType> sufOps = std::move(itm.postUnOps);

				size_t preIdx=0;
				size_t sufIdx=0;
				for (const bool isSufOp : order)
					desugarUnOp(itm, preOps, sufOps, isSufOp ? (sufIdx++) : (preIdx++), isSufOp);
			}
			return false;
		}; 
	};

	export std::vector<InlineModule> basicDesugar(parse::BasicMpDbData& mpDbData,parse::ParsedFile& itm)
	{
		DesugarVisitor vi{ {},parse::BasicMpDb{ &mpDbData } };

		visit::visitFile(vi, itm);

		return std::move(vi.inlineModules);
	}
}