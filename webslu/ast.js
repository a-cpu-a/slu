// ============================================================================
// BASE CLASSES
// ============================================================================

/// Use this when you dont need a preSpace
class CompoundNode {
    constructor(type) {
        this.type = type;
    }
}
class Node extends CompoundNode {
    constructor(type) {
        super(type)
        this.preSpace = "";
    }
}

class Token extends Node {
    constructor(txt) {
        super("Token");
        this.txt = txt;
    }
}

class Safety extends Node {
    // kind: "default" | "unsafe" | "safe"
    constructor() {
        super("Safety");
        this.kind = "default";
    }
}

class Export extends CompoundNode {
    constructor() {
        super("Export");
        this.kw = new OptToken("ex");
    }
}

class Name extends Node {
    constructor() {
        super("Name");
        this.name = "";
    }
}
class Num extends Node {
    constructor(num) {
        super("Num");
        this.num = num; // Including 0x ... p+...
    }
}
class Str extends Node {
    constructor(str) {
        super("Str");
        this.str = str; // Including the quotes or [[]]
    }
}
class TuplableName extends CompoundNode {
    constructor(type) { super(type); }
}
class NameTuplableName extends TuplableName {
    constructor() {
        super("NameTuplableName");
        this.name = new Name();
    }
}
class NumTuplableName extends TuplableName {
    constructor() {
        super("NumTuplableName");
        this.name = new Num();
    }
}

// ============================================================================
// HELPERS
// ============================================================================

class DelimitedListItem extends CompoundNode {
    constructor() {
        super("DelimitedListItem");
        this.value = null; // The actual node (e.g., Expr, TypedParam)
        this.sep = new OptToken(); // "," or ";"
    }
}

class DelimitedList extends CompoundNode {
    constructor(type) {
        super("DelimitedList");
        this.itemType = type;
        this.items = []; // Array of DelimitedListItem
    }
}

class OptToken extends Node {
    constructor(txt) {
        super("OptToken");
        this.txt = txt;
        this.present = false; // true if parsed
    }
}

class OptExpr extends CompoundNode {
    constructor() {
        super("OptExpr");
        this.expr = new Expr();
        this.present = false; // true if parsed
    }
}

class OptBlockNode extends CompoundNode {
    constructor() {
        super("OptBlockNode");
        this.present = false; // true if block exists
        this.block = new BlockNode();
    }
}
class OptTableConstructor extends CompoundNode {
    constructor() {
        super("OptTableConstructor");
        this.present = false; // true if it exists
        this.tbl = new TableConstructor();
    }
}

// ============================================================================
// ANNOTATIONS
// ============================================================================

class Annotation extends CompoundNode {
    constructor(type) { super(type); }
}

class ModPathAnnotation extends Annotation {
    constructor() {
        super("ModPathAnnotation");
        this.at = new Token("@");
        this.path = new ModPath();
        this.table = new OptTableConstructor();
    }
}

class DocLineAnnotation extends Annotation {
    constructor() {
        super("DocLineAnnotation");
        this.txt = new Token("---");
        this.content = "";
    }
}

class OuterAnnotation extends CompoundNode {
    constructor(type) { super(type); }
}

class OuterModPathAnnotation extends OuterAnnotation {
    constructor() {
        super("OuterModPathAnnotation");
        this.at = new Token("@<");
        this.path = new ModPath();
        this.table = new OptTableConstructor();
    }
}

class OuterDocLineAnnotation extends OuterAnnotation {
    constructor() {
        super("OuterDocLineAnnotation");
        this.txt = new Token("--<");
        this.content = "";
    }
}

// ============================================================================
// BLOCK OR RETURN (Helper for IfStat)
// ============================================================================

// blockOrRet ::= retstat [";"] | "{" block "}"
class BlockOrRet extends CompoundNode {
    constructor(type) { super(type); }
}

class BlockOrRetBlock extends BlockOrRet {
    constructor() {
        super("BlockOrRetBlock");
        this.block = new BlockNode();
    }
}

class BlockOrRetRetStat extends BlockOrRet {
    constructor() {
        super("BlockOrRetRetStat");
        this.retStat = new RetStat();
        this.semicol = new OptToken(";");
    }
}

// ============================================================================
// PATTERNS (sPat, dPat, uncondDestrPat)
// ============================================================================

class DestrSpec extends CompoundNode {
    constructor(type) { super(type); }
}
class SimplePatDestrSpec extends DestrSpec {
    constructor() {
        super("SimplePatDestrSpec");
        this.type = new SimplePat();
    }
}
class OpDestrSpec extends DestrSpec {
    constructor() {
        super("OpDestrSpec");
        this.mutKw = new OptToken("mut");
        this.ops = []; // Array of TypePreOp
    }
}
class Pat extends CompoundNode {
    constructor(type) { super(type); }
}
class SimplePat extends Pat {
    constructor() {
        super("SimplePat");
        this.expr = new Expr();
    }
}
class OptSimplePat extends CompoundNode {
    constructor() {
        super("OptSimplePat");
        this.present = false; // true if it exists
        this.val = new SimplePat();
    }
}
class UncondDestrPat extends Pat {
    constructor(type) { super(type); }
}
class AlwaysDestrPat extends UncondDestrPat {
    constructor() {
        super("AlwaysDestrPat");
        this.us = new Token("_");
    }
}
class UncondVarDestrPat extends UncondDestrPat {
    constructor() {
        super("UncondVarDestrPat");
        this.specifiers = new DestrSpec();
        this.name = new Name();
    }
}
class UncondPatFieldDestrPat extends UncondDestrPat {
    constructor() {
        super("UncondPatFieldDestrPat");
        this.type = new OptSimplePat();
        this.openBrace = new Token("{");
        this.fields = new DelimitedList("UncondFieldDestrField");
        this.extraFields = new OptToken("..");
        this.closeBrace = new Token("}");
    }
}
class UncondFieldDestrPat extends UncondDestrPat {
    constructor() {
        super("UncondFieldDestrPat");
        this.type = new OptSimplePat();
        this.openBrace = new Token("{");
        this.fields = new DelimitedList("UncondPatFieldDestrPat");
        this.extraFields = new OptToken("..");
        this.closeBrace = new Token("}");
    }
}
class UncondFieldDestrField extends CompoundNode {
    constructor() {
        super("UncondFieldDestrField");
        this.openPipe = new Token("|");
        this.var = new TuplableName();
        this.closePipe = new Token("|");
        this.pat = new UncondDestrPat();
    }
}

class DestrPat extends UncondDestrPat {
    constructor(type) { super(type); }
}
class VarDestrPat extends DestrPat {
    constructor() {
        super("VarDestrPat");
        this.base = new UncondVarDestrPat();
        this.eq = new OptToken("=");
        this.valPat = new SimplePat(); // $<Needs eq>
    }
}
class PatFieldDestrPat extends DestrPat {
    constructor() {
        super("PatFieldDestrPat");
        this.type = new OptSimplePat();
        this.openBrace = new Token("{");
        this.fields = new DelimitedList("Pat");
        this.extraFields = new OptToken("..");
        this.closeBrace = new Token("}");
    }
}
class FieldDestrPat extends DestrPat {
    constructor() {
        super("FieldDestrPat");
        this.type = new OptSimplePat();
        this.openBrace = new Token("{");
        this.fields = new DelimitedList("FieldDestrField");
        this.extraFields = new OptToken("..");
        this.closeBrace = new Token("}");
    }
}
class FieldDestrField extends CompoundNode {
    constructor() {
        super("FieldDestrField");
        this.openPipe = new Token("|");
        this.var = new TuplableName();
        this.closePipe = new Token("|");
        this.pat = new Pat();
    }
}

// ============================================================================
// PREOPS
// ============================================================================

class RefAttrs extends CompoundNode {
    constructor() {
        super("RefAttrs");
        this.addrspace = null; // OptToken("in" Name)
        this.lifetime = null;  // OptLifetime
        this.refType = new RefType();
    }
}
class RefType extends CompoundNode {
    constructor(type) { super(type); }
}
class MutRefType extends RefType {
    constructor() { super("MutRefType"); this.kw = new Token("mut"); }
}
class ShareRefType extends RefType {
    constructor() { super("ShareRefType"); this.kw = new Token("share"); }
}
class ConstRefType extends RefType {
    constructor() { super("ConstRefType"); this.kw = new Token("const"); }
}

class PreOp extends CompoundNode {
    constructor(type) { super(type); }
}
class TypePreOp extends PreOp {
    constructor(type) { super(type); }
}

class RefTypePreOp extends TypePreOp {
    constructor() {
        super("RefTypePreOp");
        this.star = new Token("*");
        this.attrs = new RefAttrs();
    }
}

class SlicePreOp extends PreOp {
    constructor() {
        super("SlicePreOp");
        this.open = new Token("[");
        this.close = new Token("]");
    }
}
class DynPreOp extends PreOp {
    constructor() { super("DynPreOp"); this.kw = new Token("dyn"); }
}
class ImplPreOp extends PreOp {
    constructor() { super("ImplPreOp"); this.kw = new Token("impl"); }
}
class UnionPreOp extends PreOp {
    constructor() { super("UnionPreOp"); this.kw = new Token("union"); }
}
class MutPreOp extends PreOp {
    constructor() { super("MutPreOp"); this.kw = new Token("mut"); }
}
class RefPreOp extends PreOp {
    constructor() {
        super("RefPreOp");
        this.amp = new Token("&");
        this.attrs = new RefAttrs();
    }
}
class RangePreOp extends PreOp {
    constructor() { super("RangePreOp"); this.op = new Token(".."); }
}
class AnnotationPreOp extends PreOp {
    constructor() {
        super("AnnotationPreOp");
        this.annotation = new Annotation();
    }
}
class IfPreOp extends PreOp {
    constructor() {
        super("IfPreOp");
        this.ifKw = new Token("if");
        this.expr = new Expr();
        this.arrow = new Token("=>");
    }
}

// ============================================================================
// EXPRESSIONS & STATEMENTS
// ============================================================================

class Stat extends CompoundNode {
    constructor(type) { super(type); }
}

class Expr extends CompoundNode {
    constructor() {
        super("Expr");
    }
}
class UnOp extends CompoundNode {
    constructor() {
        super("UnOp");
    }
}
class Args extends CompoundNode {
    constructor(type) { super(type); }
}
class TableArgs extends Args {
    constructor() {
        super("TableArgs");
        this.args = new TableConstructor();
    }
}
class ParenArgs extends Args {
    constructor() {
        super("ParenArgs");
        this.openParen = new Token("(");
        this.args = new DelimitedList("Expr");
        this.closeParen = new Token(")");
    }
}
class NumArgs extends Args {
    constructor() {
        super("NumArgs");
        this.args = new Num();
    }
}
class StrArgs extends Args {
    constructor() {
        super("StrArgs");
        this.args = new Str();
    }
}

class BlockNode extends CompoundNode {
    constructor() {
        super("BlockNode");
        this.openBrace = new Token("{");
        this.stats = [];
        this.retStat = new RetStat();
        this.semicol = new OptToken(";");
        this.closeBrace = new Token("}");
    }
}

class ModPath extends CompoundNode {
    constructor() {
        super("ModPath");
        this.root = new ModPathRoot();
        this.segments = []; // Array of ModPathSegment
    }
}

class ModPathRoot extends CompoundNode {
    constructor(type) { super(type); }
}
class ModPathRootName extends ModPathRoot {
    constructor() {
        super("ModPathRootName");
        this.name = new Name();
    }
}
class ModPathRootSelf extends ModPathRoot {
    constructor() {
        super("ModPathRootSelf");
        this.kw = new Token("self");
    }
}
class ModPathRootCrate extends ModPathRoot {
    constructor() {
        super("ModPathRootCrate");
        this.kw = new Token("crate");
    }
}
class ModPathRootOp extends ModPathRoot {
    constructor() {
        super("ModPathRootOp");
        this.op = new Token(":>");
    }
}

// "::" Name
class ModPathSegment extends CompoundNode {
    constructor() {
        super("ModPathSegment");
        this.sep = new Token("::");
        this.name = new Name();
    }
}

class MatchItem extends CompoundNode {
    constructor() {
        super("MatchItem");
        this.pat = new Pat();

        this.ifKw = new OptToken("if");
        this.ifExpr = new Expr(); // $<Needs ifKw>

        this.arrow = new Token("=>");
        this.expr = new Expr();
    }
}
class MatchTypeBlock extends CompoundNode {
    constructor() {
        super("MatchTypeBlock");

        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>

        this.openBrace = new Token("{");
        this.items = new DelimitedList("MatchItem");
        this.closeBrace = new Token("}");
    }
}

class TypedParam extends CompoundNode {
    constructor() {
        super("TypedParam");
        this.constKw = new OptToken("const");
        this.name = new Name();
        this.eq = new Token("=");
        this.type = new Expr();
    }
}

// ============================================================================
// TABLE CONSTRUCTOR
// ============================================================================

class Field extends CompoundNode {
    constructor(type) { super(type); }
}

class NamedField extends Field {
    constructor() {
        super("NamedField");
        this.name = new Name();
        this.eq = new Token("=");
        this.expr = new Expr();
    }
}

class ExprField extends Field {
    constructor() {
        super("UnnamedField");
        this.expr = new Expr();
    }
}

class TableConstructor extends Expr {
    constructor() {
        super("TableConstructor");
        this.openBrace = new Token("{");
        this.fields = new DelimitedList("Field");
        this.closeBrace = new Token("}");
    }
}

// ============================================================================
// GLOBAL DECLARATIONS (globstat)
// ============================================================================

class GlobStat extends CompoundNode {
    constructor(type) { super(type); }
}

// globstat ::= ";"
class EmptyGlobStat extends GlobStat {
    constructor() {
        super("EmptyGlobStat");
        this.semicol = new Token(";");
    }
}

// optexport "struct" Name ["(" params ")"] tableconstructor
class StructDecl extends GlobStat {
    constructor() {
        super("StructDecl");
        this.export = new Export();
        this.structKw = new Token("struct");
        this.name = new Name();

        this.openParen = new OptToken("(");
        this.params = new DelimitedList("TypedParam"); // $<Needs openParen>
        this.closeParen = new Token(")"); // $<Needs openParen>

        this.body = new TableConstructor();
    }
}

// optexport "enum" Name ["(" params ")"] "{" {enumfield fieldsep} [".."] "}"
class EnumDecl extends GlobStat {
    constructor() {
        super("EnumDecl");
        this.export = new Export();
        this.enumKw = new Token("enum");
        this.name = new Name();

        this.openParen = new OptToken("(");
        this.params = new DelimitedList("TypedParam"); // $<Needs openParen>
        this.closeParen = new Token(")"); // $<Needs openParen>

        this.openBrace = new Token("{");
        this.fields = new DelimitedList("EnumField");
        this.spread = new OptToken("..");
        this.closeBrace = new Token("}");
    }
}

class EnumField extends CompoundNode {
    constructor() {
        super("EnumField");
        this.annotations = []; // Array of Annotation
        this.export = new Export();
        this.name = new Name();
        this.table = new OptTableConstructor(); // TableConstructor
        this.outerAnnotations = []; // Array of OuterAnnotation
    }
}

// optexport [safety] ["struct"] "fn" Name "(" [params] ")" ["->" basicExpr] ["{" block "}"]
class FunctionDecl extends GlobStat {
    constructor() {
        super("FunctionDecl");
        this.export = new Export();
        this.safety = new Safety();
        this.structKw = new OptToken("struct");
        this.fnKw = new Token("fn");
        this.name = new Name();
        this.openParen = new Token("(");
        this.params = new DelimitedList("TypedParam");
        this.closeParen = new Token(")");
        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>
        this.body = new OptBlockNode();
    }
}

// optexport "trait" Name ["(" params ")"] [whereClauses] tableconstructor
class TraitDecl extends GlobStat {
    constructor() {
        super("TraitDecl");
        this.export = new Export();
        this.traitKw = new Token("trait");
        this.name = new Name();

        this.openParen = new OptToken("(");
        this.params = new DelimitedList("TypedParam"); // $<Needs openParen>
        this.closeParen = new Token(")"); // $<Needs openParen>

        this.where = new WhereClauses();
        this.body = new TableConstructor();
    }
}

// safety extern LiteralString "{" {globstat} "}"
class ExternBlock extends GlobStat {
    constructor() {
        super("ExternBlock");
        this.safety = new Safety();
        this.externKw = new Token("extern");
        this.abiName = new Str();
        this.openBrace = new Token("{");
        this.stats = []; // Array of GlobStat
        this.closeBrace = new Token("}");
    }
}

// optexport ["unsafe"] "impl" ["(" params ")"] [basicExpr "for"] basicExpr [whereClauses] tableconstructor
class ImplDecl extends GlobStat {
    constructor() {
        super("ImplDecl");
        this.export = new Export();
        this.safety = new Safety();
        this.implKw = new Token("impl");

        this.openParen = new OptToken("(");
        this.params = new DelimitedList("TypedParam"); // $<Needs openParen>
        this.closeParen = new Token(")"); // $<Needs openParen>

        this.traitType = new Expr(); // $<Needs forKw>
        this.forKw = new OptToken("for");

        this.targetType = new Expr();

        this.where = new WhereClauses();
        this.body = new TableConstructor();
    }
}

// optexport "use" modpath useVariant
class UseDecl extends GlobStat {
    constructor() {
        super("UseDecl");
        this.export = new Export();
        this.useKw = new Token("use");
        this.path = new ModPath();
        this.variant = new UseVariant();
    }
}

class UseVariant extends CompoundNode {
    constructor(type) { super(type); }
}

class UseAs extends UseVariant {
    constructor() {
        super("UseAs");
        this.asKw = new Token("as");
        this.name = new Name();
    }
}

class SimpleUseVariant extends UseVariant {
    constructor() {
        super("SimpleUseVariant");
    }
}

// "::*"
class StarUseVariant extends UseVariant {
    constructor() {
        super("StarUseVariant");
        this.kw = new Token("::*");
    }
}

// "::{" ("self"|Name) {fieldsep Name} [fieldsep] "}"
class BraceUseVariant extends UseVariant {
    constructor() {
        super("BraceUseVariant");
        this.colonColon = new Token("::");
        this.openBrace = new Token("{");

        this.selfKw = new OptToken("self");
        this.selfDelim = new Token(""); // $<Needs selfKw>

        this.items = new DelimitedList("Name");
        this.closeBrace = new Token("}");
    }
}

// optexport "mod" Name ["{" chunk "}"]
class ModDecl extends GlobStat {
    constructor() {
        super("ModDecl");
        this.export = new Export();
        this.modKw = new Token("mod");
        this.name = new Name();
        this.openBrace = new OptToken("{");
        this.chunk = []; // Array of globstat // $<Needs openBrace>
        this.closeBrace = new Token("}"); // $<Needs openBrace>
    }
}

// optexport "const" uncondDestrPat "=" expr
class ConstDecl extends GlobStat {
    constructor() {
        super("ConstDecl");
        this.export = new Export();
        this.constKw = new Token("const");
        this.pattern = new UncondDestrPat();
        this.eq = new Token("=");
        this.value = new Expr();
    }
}

// optexport "union" Name ["(" params ")"] tableconstructor
class UnionDecl extends GlobStat {
    constructor() {
        super("UnionDecl");
        this.export = new Export();
        this.unionKw = new Token("union");
        this.name = new Name();

        this.openParen = new OptToken("(");
        this.params = new DelimitedList("TypedParam"); // $<Needs openParen>
        this.closeParen = new Token(")"); // $<Needs openParen>

        this.body = new TableConstructor();
    }
}

// ============================================================================
// STATEMENTS (stat)
// ============================================================================

class RetStat extends CompoundNode {
    constructor(type) { super(type); }
}

class ReturnStat extends RetStat {
    constructor() {
        super("ReturnStat");
        this.returnKw = new Token("return");
        this.expr = new OptExpr();
    }
}

class BreakStat extends RetStat {
    constructor() {
        super("BreakStat");
        this.breakKw = new Token("break");
        this.label = new OptToken("'");
        this.labelName = new Name(); // $<Needs label>
        this.expr = new OptExpr();
    }
}

class ContinueStat extends RetStat {
    constructor() {
        super("ContinueStat");
        this.continueKw = new Token("continue");
        this.label = new OptToken("'");
        this.labelName = new Name(); // $<Needs label>
        this.expr = new OptExpr();
    }
}

class ThrowStat extends RetStat {
    constructor() {
        super("ThrowStat");
        this.throwKw = new Token("throw");
        this.expr = new Expr();
    }
}

// [label] "loop" ["->" basicExpr] "{" block "}"
class LoopStat extends Stat {
    constructor() {
        super("LoopStat");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>

        this.loopKw = new Token("loop");
        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>
        this.body = new BlockNode();
    }
}

// [label] "while" basicExpr "{" block "}"
class WhileStat extends Stat {
    constructor() {
        super("WhileStat");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>

        this.whileKw = new Token("while");
        this.condition = new Expr();
        this.body = new BlockNode();
    }
}

// [label] "for" ["const"] uncondDestrPat "in" basicExpr "{" block "}"
class ForStat extends Stat {
    constructor() {
        super("ForStat");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>

        this.forKw = new Token("for");
        this.constKw = new OptToken("const");
        this.pattern = new UncondDestrPat();
        this.inKw = new Token("in");
        this.iterable = new Expr();
        this.body = new BlockNode();
    }
}

// "let" uncondDestrPat ["=" expr]
class LetStat extends Stat {
    constructor() {
        super("LetStat");
        this.letKw = new Token("let");
        this.pattern = new UncondDestrPat();
        this.eq = new OptToken("=");
        this.value = new Expr(); // $<Needs eq>
    }
}

// Represents the "else if ... " part of an IfStat
class IfAlternate extends CompoundNode {
    constructor() {
        super("IfAlternate");
        this.elseKw = new Token("else");
        this.ifKw = new Token("if");
        this.condition = new Expr();
        this.body = new BlockOrRet();
    }
}
// "if" basicExpr blockOrRet {"else" "if" basicExpr blockOrRet} ["else" blockOrRet]
class IfStat extends Stat {
    constructor() {
        super("IfStat");
        this.ifKw = new Token("if");
        this.condition = new Expr();
        this.consequent = new BlockOrRet();
        this.alternates = []; // Array of IfAlternate
        this.elseKw = new OptToken("else");
        this.elseBlock = new BlockOrRet();
    }
}

// "match" basicExpr matchtypeblock
class MatchStat extends Stat {
    constructor() {
        super("MatchStat");
        this.matchKw = new Token("match");
        this.argument = new Expr();
        this.block = new MatchTypeBlock();
    }
}

// "drop" expr
class DropStat extends Stat {
    constructor() {
        super("DropStat");
        this.dropKw = new Token("drop");
        this.expr = new Expr();
    }
}

// [label] "{" block "}"
class BlockStat extends Stat {
    constructor() {
        super("BlockStat");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>
        this.block = new BlockNode();
    }
}

// ";"
class EmptyStat extends Stat {
    constructor() {
        super("EmptyStat");
        this.semicol = new Token(";");
    }
}

// "unsafe" "{" {stat} "}"
class UnsafeStat extends Stat {
    constructor() {
        super("UnsafeStat");
        this.unsafeKw = new Token("unsafe");
        this.openBrace = new Token("{");
        this.stats = []; // Array of Stat
        this.closeBrace = new Token("}");
    }
}

// var "=" expr
class AssignStat extends Stat {
    constructor() {
        super("AssignStat");
        this.var = new Var();
        this.eq = new Token("=");
        this.expr = new Expr();
    }
}

// var selfablecall
class CallStat extends Stat {
    constructor() {
        super("CallStat");
        this.var = new Var();
        this.call = new SelfableCall();
    }
}

// "TODO!" LiteralString
class TodoStat extends Stat {
    constructor() {
        super("TodoStat");
        this.kw = new Token("TODO!");
        this.msg = new Expr();
    }
}

// ============================================================================
// EXPRESSIONS (expr, basicExpr)
// ============================================================================

// "(" expr ")"
class ParenExpr extends Expr {
    constructor() {
        super("ParenExpr");
        this.openParen = new Token("(");
        this.expr = new Expr();
        this.closeParen = new Token(")");
    }
}

class ModPathExpr extends Expr {
    constructor() {
        super("ModPathExpr");
        this.path = new ModPath();
    }
}

class StrExpr extends Expr {
    constructor() {
        super("StrExpr");
        this.raw = new Str();
    }
}

class NumExpr extends Expr {
    constructor() {
        super("NumExpr");
        this.raw = new Num();
    }
}

class BinExpr extends Expr {
    constructor() {
        super("BinExpr");
        this.left = new Expr();
        this.op = new Token("");
        this.right = new Expr();
    }
}

class UnaryExpr extends Expr {
    constructor() {
        super("UnaryExpr");
        this.PreOps = []; // Array of PreOp
        this.primary = new Expr();
        this.sufOps = []; // Array of SufOp
    }
}

class SufOp extends CompoundNode {
    constructor(type) { super(type); }
}

// var ::= (Name | "(" expr ")") {{selfablecall} subvar}
class Var extends CompoundNode {
    constructor() {
        super("Var");
        this.root = null; // Name or ParenExpr
        this.suffixes = []; // Array of SubVar | SelfableCall
    }
}

class SubVar extends SufOp {
    constructor() { super("SubVar"); }
}
class DerefSubVar extends SubVar {
    constructor() { super("DerefSubVar"); this.op = new Token(".*"); }
}
class DotSubVar extends SubVar {
    constructor() { super("DotSubVar"); this.op = new Token("."); this.field = new TuplableName(); }
}
class ConstDotSubVar extends SubVar {
    constructor() { super("ConstDotSubVar"); this.op = new Token(".:"); this.field = new Name(); }
}
class IdxSubVar extends SubVar {
    constructor() { super("IdxSubVar"); this.open = new Token("["); this.expr = new Expr(); this.close = new Token("]"); }
}

class SelfableCall extends SufOp {
    constructor() {
        super("SelfableCall");
        this.dot = new OptToken(".");
        this.method = new Name(); // $<Needs dot>
        this.args = new Args();
    }
}

class TryOp extends SufOp {
    constructor() {
        super("TryOp");
        this.tryKw = new Token("try");
        this.block = new MatchTypeBlock();
    }
}

class MatchExpr extends Expr {
    constructor() {
        super("MatchExpr");
        this.matchKw = new Token("match");
        this.argument = new Expr();
        this.block = new MatchTypeBlock();
    }
}
// [label] "loop" ["->" basicExpr] "{" block "}"
class LoopExpr extends Expr {
    constructor() {
        super("LoopExpr");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>

        this.loopKw = new Token("loop");
        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>
        this.body = new BlockNode();
    }
}

class FnExpr extends Expr {
    constructor() {
        super("FnExpr");
        this.safety = new Safety();
        this.fnKw = new Token("fn");
        this.openParen = new Token("(");

        this.selfParamRefs = []; // Array of RefTypePreOp // $<Needs selfParam>
        this.selfParam = new OptToken("self");
        this.selfParamDelim = new Token(""); // $<Needs selfParam>

        this.params = new DelimitedList("TypedParam");
        this.closeParen = new Token(")");
        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>
        this.body = new OptBlockNode();
    }
}

class LambdaExpr extends Expr {
    constructor() {
        super("LambdaExpr");
        this.safety = new Safety();
        this.pipe1 = new Token("|");
        this.params = new DelimitedList("TypedParam");
        this.pipe2 = new Token("|");

        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>
        this.doubleArrow = new Token("=>"); // $<Needs retArrow>

        this.body = new Expr();
    }
}

// [label] ["const"] "do" ["->" basicExpr] "{" block "}"
class DoExpr extends Expr {
    constructor() {
        super("DoExpr");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>

        this.constKw = new OptToken("const");
        this.doKw = new Token("do");

        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>

        this.block = new BlockNode();
    }
}

class ConstExpr extends Expr {
    constructor() {
        super("ConstExpr");
        this.constKw = new Token("const");
        this.openParen = new Token("(");
        this.expr = new Expr();
        this.closeParen = new Token(")");
    }
}

class TodoExpr extends Expr {
    constructor() {
        super("TodoExpr");
        this.kw = new Token("TODO!");
        this.msg = new Str();
    }
}

class UnderscoreExpr extends Expr {
    constructor() {
        super("UnderscoreExpr");
        this.us = new Token("_");
    }
}

class LifetimeExpr extends Expr {
    constructor() {
        super("LifetimeExpr");
        this.names = []; // Array of {kw:Token("/"),l:Name}
    }
}

class UnboundedRangeExpr extends Expr {
    constructor() {
        super("UnboundedRangeExpr");
        this.op = new Token("..");
    }
}

// "_COMP_TODO!" "(" LiteralString "," expr ")"
class CompTodoExpr extends Expr {
    constructor() {
        super("CompTodoExpr");
        this.kw = new Token("_COMP_TODO!");
        this.openParen = new Token("(");
        this.msg = new Str(); // LiteralString
        this.comma = new Token(",");
        this.expr = new Expr();
        this.closeParen = new Token(")");
    }
}

// ============================================================================
// UTILS
// ============================================================================

class WhereClauses extends CompoundNode {
    constructor() {
        super("WhereClauses");
        this.whereKw = new Token("where");
        this.clauses = new DelimitedList("WhereClause");
    }
}

class WhereClause extends CompoundNode {
    constructor() {
        super("WhereClause");
        this.name = new Name(); // or "Self"
        this.colon = new Token(":");
        this.type = new Expr();
    }
}







/**
 * Visitor implementation for the provided AST.
 * 
 * Features:
 * 1. Generic Traversal: Automatically visits all child nodes.
 * 2. Mutation: Supports replacing nodes in-place or returning new nodes.
 * 3. Scope Management: Built-in stack for symbol tables (useful for name resolution).
 * 
 * Extend this class to create specific passes (e.g., TypeChecker, ConstEvaluator).
 */
class Visitor {
    constructor() {
        // Stack of scopes for symbol resolution
        this.scopeStack = [];
    }

    // ============================================================
    // MAIN ENTRY POINT
    // ============================================================

    /**
     * Visits an AST node.
     * If a specific method 'visit<NodeType>' exists, it is called.
     * Otherwise, defaults to generic traversal of children.
     * 
     * @param {Node|CompoundNode|null} node 
     * @returns {Node|CompoundNode|null} The original node, a mutated node, or a replacement node.
     */
    visit(node) {
        if (!node) return null;

        // Only process objects (Nodes)
        if (typeof node !== 'object') return node;

        // Dispatch to specific handler if defined
        const methodName = 'visit' + node.type;
        if (this[methodName] && typeof this[methodName] === 'function') {
            const result = this[methodName](node);
            // If the visitor returns a value, assume it is the new node (mutation/replacement).
            // If it returns undefined, assume it mutated in-place and return the original node.
            return result !== undefined ? result : node;
        }

        // Default behavior: recurse into children
        return this.visitChildren(node);
    }

    // ============================================================
    // GENERIC TRAVERSAL ENGINE
    // ============================================================

    /**
     * Recursively visits all properties of a node.
     * Handles arrays (of nodes) and single node references.
     * It attempts to update the parent references if a child node is replaced.
     */
    visitChildren(node) {
        if (!node || typeof node !== 'object') return node;

        for (const key in node) {
            if (!node.hasOwnProperty(key)) continue;

            // Skip metadata and helper properties
            if (key === 'type' || key === 'preSpace' || key === 'itemType') continue;

            const value = node[key];

            // 1. Handle Arrays (e.g., stats, params, items)
            if (Array.isArray(value)) {
                for (let i = 0; i < value.length; i++) {
                    const item = value[i];
                    // Check if item is an AST Node
                    if (item && (item instanceof Node || item instanceof CompoundNode)) {
                        const newItem = this.visit(item);
                        // Mutation Support: Replace array item if the visitor returned a new object
                        if (newItem !== item) {
                            value[i] = newItem;
                        }
                    }
                }
            }
            // 2. Handle Single Nodes (e.g., expr, body, condition)
            else if (value && (value instanceof Node || value instanceof CompoundNode)) {
                // Optimization: If the node is an "Opt" wrapper (OptExpr, OptToken, etc.)
                // and it is not present, skip visiting its inner dummy node.
                if (value.hasOwnProperty('present') && value.present === false) {
                    continue;
                }

                const newValue = this.visit(value);
                // Mutation Support: Replace property if the visitor returned a new object
                if (newValue !== value) {
                    node[key] = newValue;
                }
            }
        }
        return node;
    }

    // ============================================================
    // SCOPE & SYMBOL MANAGEMENT HELPERS
    // ============================================================

    /** Returns the current active scope */
    get currentScope() {
        return this.scopeStack[this.scopeStack.length - 1];
    }

    /** Enters a new scope (e.g., entering a Block or Function) */
    enterScope(node) {
        const scope = {
            node: node,
            parent: this.currentScope,
            symbols: {}, // name -> { type, node, kind }
            types: {}    // name -> TypeNode
        };
        this.scopeStack.push(scope);
    }

    /** Exits the current scope */
    leaveScope() {
        if (this.scopeStack.length > 0) {
            this.scopeStack.pop();
        }
    }

    /** Defines a symbol in the current scope */
    defineSymbol(name, metadata) {
        if (this.currentScope) {
            this.currentScope.symbols[name] = metadata;
        } else {
            console.warn(`Defining symbol ${name} outside of a scope`);
        }
    }

    /** Looks up a symbol in the scope stack */
    resolveSymbol(name) {
        for (let i = this.scopeStack.length - 1; i >= 0; i--) {
            const scope = this.scopeStack[i];
            if (scope.symbols[name]) {
                return scope.symbols[name];
            }
        }
        return null;
    }

    // ============================================================
    // BASE STRUCTURAL VISITORS (OVERRIDE FOR SCOPE LOGIC)
    // ============================================================

    // Override these to inject scope management automatically

    visitBlockNode(node) {
        this.enterScope(node);
        this.visitChildren(node);
        this.leaveScope();
        return node;
    }

    visitFunctionDecl(node) {
        this.enterScope(node);
        this.visitChildren(node);
        this.leaveScope();
        return node;
    }

    visitImplDecl(node) {
        this.enterScope(node);
        this.visitChildren(node);
        this.leaveScope();
        return node;
    }

    visitForStat(node) {
        this.enterScope(node);
        this.visitChildren(node);
        this.leaveScope();
        return node;
    }

    // ============================================================
    // EXAMPLE: CONST EVALUATOR / INLINING
    // ============================================================

    /*
    // Example of how to extend the visitor for Const Inlining
	
    visitModPathExpr(node) {
        const pathStr = this.modPathToString(node.path);
        const sym = this.resolveSymbol(pathStr);
    	
        // If symbol is a constant and has a known evaluated value
        if (sym && sym.kind === 'const' && sym.value) {
            // Return the value node directly (Inlining)
            // We clone it to avoid mutating the original definition
            return this.cloneNode(sym.value);
        }
    	
        return node;
    }
	
    visitBinExpr(node) {
        this.visitChildren(node);
    	
        // Simple constant folding example: 1 + 2 -> 3
        if (node.left.type === 'NumExpr' && node.right.type === 'NumExpr') {
            if (node.op.txt === '+') {
                const l = parseFloat(node.left.raw.num);
                const r = parseFloat(node.right.raw.num);
                const newNode = new NumExpr(new Num((l + r).toString()));
                newNode.preSpace = node.preSpace;
                return newNode;
            }
        }
        return node;
    }
    */
}



// ============================================================================
// PARSER IMPLEMENTATION
// ============================================================================

class Parser {
    constructor(input) {
        this.input = input;
        this.pos = 0;
        this.len = input.length;
        this.tokens = [];
        this.tokPos = 0;
        this.tokenize();
    }

    // ========================================================================
    // LEXER / TOKENIZER
    // ========================================================================

    tokenize() {
        // Define Keywords based on spec
        const keywords = new Set([
            "drop", "trait", "any", "has", "raw", "glob", "reloc", "nosize", "concept", "nostride",
            "unstrided", "at", "of", "asm", "box", "out", "auto", "case", "only", "pure", "test",
            "with", "wrap", "final", "inout", "mixin", "become", "impure", "Inject", "inject",
            "inline", "typeof", "Modify", "Shadow", "Capture", "default", "discard", "uniform",
            "unsized", "virtual", "abstract", "override", "Operation", "operation", "Overwrite",
            "overwrite", "groupshare", "groupshared", "threadlocal", "gen", "copy", "move", "async",
            "await", "super", "yield", "static", "generator", "as", "ex", "fn", "it", "dyn", "let",
            "mod", "mut", "rec", "try", "use", "also", "enum", "impl", "loop", "safe", "self",
            "Self", "TODO", "alloc", "axiom", "catch", "const", "crate", "defer", "macro", "match",
            "share", "throw", "trans", "union", "where", "extern", "struct", "unsafe", "continue",
            "_COMP_TODO", "do", "if", "in", "or", "and", "end", "for", "else", "goto", "then",
            "break", "local", "until", "while", "elseif", "global", "repeat", "return", "function"
        ]);

        // Define Symbols based on spec
        // Using a Set for O(1) lookup. We will check for length 3, then 2, then 1.
        const symbols = new Set([
            "|||", ">=<", "===", "<=>", "+++", "//=", "///", "***", "**-", "...", ":::", "---",
            "~~", "~|", "~>", "~=", "~<", "~+", "~^", "~%", "~&", "~/", "~@", "~?", "~!", "~-", "~",
            "||", "|>", "|=", "|<", "|+", "|^", "|%", "|&", "|/", "|@", "|?", "|!", "|-", "|",
            ">~", ">|", ">>", ">=", "><", ">+", ">^", ">%", ">&", ">/", ">*", ">@", ">?", ">!", ">",
            "=<", "=",
            "<~", "<|", "<>", "<=", "<<", "<+", "<^", "<%", "<&", "</", "<*", "<@", "<?", "<!", "<",
            "+~", "+|", "+>", "+=", "+<", "++", "+^", "+%", "+/", "+*", "+@", "+!", "+-", "+",
            "^~", "^|", "^>", "^=", "^<", "^+", "^^", "^%", "^/", "^*", "^@", "^?", "^!", "^",
            "%~", "%|", "%>", "%=", "%<", "%+", "%^", "%%", "%/", "%*", "%@", "%?", "%!", "%",
            "&~", "&>", "&=", "&<", "&+", "&^", "&%", "&&", "&*", "&?", "&",
            "/~", "/|", "/>", "/=", "/<", "/+", "/^", "/%", "/&", "//", "/*", "/@", "/?", "/!", "/",
            "*~", "*|", "*>", "*=", "*<", "*+", "*^", "*%", "*/", "**", "*?", "*!", "*",
            "@<", "@@", "@", "}", "{", "]", "[", ")", "(", "\"", "'", ".*", "..", ".:", ".", "?&", "?@", "?!", "?",
            "!~", "!|", "!>", "!=", "!<", "!+", "!^", "!%", "!/", "!*", "!@", "!?", "!!", "!",
            ":>", ":",
            ";", ",",
            "-~", "-|", "->", "-=", "-<", "-+", "-^", "-%", "-/", "-*", "-@", "-?", "--", "-",
            "__", "_"
        ]);

        // Helper to handle Lua-style long brackets
        const handleLongBracket = (isComment) => {
            const start = this.pos;
            this.pos++; // Skip '['

            const openBracketPos = this.pos;
            let level = 0;

            // Count '=' signs
            while (this.pos < this.len && this.input[this.pos] === '=') {
                this.pos++;
                level++;
            }

            // Expect closing '['
            if (this.pos >= this.len || this.input[this.pos] !== '[') {
                this.pos = start; // Reset and let standard logic handle it
                if (isComment)
                    throw new Error(`Expected second bracket for multi line comment: ${this.input[this.pos]} at ${this.pos}`);
                return false;
            }
            this.pos++; // Consume the second '['

            const openSeq = this.input.substring(openBracketPos - 1, this.pos);
            const closeSeq = openSeq.replace(/\[/g, ']'); // [==[ becomes ]==]

            // Scan for closing sequence
            while (this.pos + closeSeq.length <= this.len) {
                if (this.input.substring(this.pos, this.pos + closeSeq.length) === closeSeq) {
                    this.pos += closeSeq.length;
                    return true; // Success
                }
                this.pos++;
            }
            this.pos = this.len;
            //TODO: soft error
            throw new Error(`Expected closing bracket for ${isComment ? "comment" : "string"}, but file ended`);
        };

        let savedStart = null;
        let lastNonSpace = 0;
        while (this.pos < this.len) {
            const start = (savedStart != null) ? savedStart : this.pos;
            savedStart = null;

            let ch = this.input[this.pos];
            let preSpace = "";

            // Skip whitespace and capture it
            while (/\s/.test(ch) && this.pos < this.len) {
                this.pos++;
                ch = this.input[this.pos];
            }
            if (start !== this.pos) {
                preSpace = this.input.substring(start, this.pos);
            }

            if (this.pos >= this.len) break; // TODO: final preSpace must be stored in the ast, prob a new field in the file or something

            const tokenStart = this.pos;

            // Comments: --...
            if (ch === '-' && this.pos + 1 < this.len && this.input[this.pos + 1] === '-') {
                this.pos += 2;
                const third = (this.pos < this.len) ? this.input[this.pos] : null;
                // Doc Comments / LineOfText: --- or --<
                if (third === '-' || third === '<') {
                    this.pos++;
                    this.tokens.push({
                        type: 'Symbol',
                        txt: "--" + third,
                        preSpace: preSpace
                    });
                    if (this.pos < this.len) {
                        ch = this.input[this.pos];
                        // Long String Literal: [[...]] or [=[...]=]
                        if (ch === '[') {
                            // We check if next char is [ or =
                            if (this.pos + 1 < this.len && (this.input[this.pos + 1] === '[' || this.input[this.pos + 1] === '=')) {
                                if (handleLongBracket(true)) {
                                    this.tokens.push({
                                        type: 'LiteralString',
                                        txt: this.input.substring(tokenStart + 3, this.pos),
                                        preSpace: ""
                                    });
                                    lastNonSpace = this.pos;
                                    continue;
                                }
                            }
                            // If not a long bracket (or malformed), fall through to symbol handling
                        }
                    }

                    let content = "";
                    while (this.pos < this.len && !/[\r\n]/.test(this.input[this.pos])) {
                        content += this.input[this.pos];
                        this.pos++;
                    }
                    this.tokens.push({
                        type: 'LineOfText',
                        txt: content,
                        preSpace: ""
                    });
                    lastNonSpace = this.pos;
                    continue;
                }
                // Multiline Comment: --[=[...]=]
                if (third === '[') {
                    if (this.pos + 1 < this.len && (this.input[this.pos + 1] === '[' || this.input[this.pos + 1] === '=')) {
                        if (handleLongBracket(true)) {
                            // Comment is whitespace, so we skip it (do not push token)

                            // Preserve preSpace for the next token
                            savedStart = start;
                            continue;
                        }
                    }
                    // Treat as line comment start if bracket invalid.
                }

                // Regular Single Line Comment: --
                while (this.pos < this.len && !/[\r\n]/.test(this.input[this.pos])) {
                    this.pos++;
                }
                // Preserve preSpace for the next token
                savedStart = start;

                // Whitespace is ignored, so we just continue
                continue;
            }

            // Strings
            if (ch === '"') {
                const quote = ch;
                this.pos++;
                while (this.pos < this.len && this.input[this.pos] !== quote) {
                    if (this.input[this.pos] === '\\') this.pos++; // skip escape
                    this.pos++;
                }
                if (this.pos >= this.len)
                    throw new Error('Expected end quote for string, but file ended');
                this.pos++; // consume end quote
                this.tokens.push({ type: 'LiteralString', txt: this.input.substring(tokenStart, this.pos), preSpace: preSpace });
                lastNonSpace = this.pos;
                continue;
            }

            // Long String Literal: [[...]] or [=[...]=]
            if (ch === '[') {
                // We check if next char is [ or =
                if (this.pos + 1 < this.len && (this.input[this.pos + 1] === '[' || this.input[this.pos + 1] === '=')) {
                    if (handleLongBracket(false)) {
                        this.tokens.push({
                            type: 'LiteralString',
                            txt: this.input.substring(tokenStart, this.pos),
                            preSpace: preSpace
                        });
                        lastNonSpace = this.pos;
                        continue;
                    }
                }
                // If not a long bracket (or malformed), fall through to symbol handling
            }
            {
                let tmpFound = false;
                // Symbols
                for (let i = 3; i > 0; i--) {
                    if ((this.pos + i - 1) < this.len) {
                        const tok = this.input.substring(this.pos, this.pos + i);
                        if (symbols.has(tok)) {
                            this.pos += i;
                            this.tokens.push({ type: 'Symbol', txt: tok, preSpace: preSpace });
                            tmpFound = true;
                            break;
                        }
                    }
                }
                if (tmpFound) {
                    lastNonSpace = this.pos;
                    continue;
                }
            }

            // Numbers (Hex or Dec)
            if (/[0-9]/.test(ch)) {
                let tokenStart = this.pos;
                let isHex = false;

                // Helper to parse digit lists based on grammar: digit [{digit|"_"}digit]
                // Returns true if at least one digit was consumed.
                const parseList = (hexMode) => {
                    const digitOrUscoreRegex = hexMode ? /[0-9_a-fA-F]/ : /[0-9_]/;
                    const start = this.pos;
                    let hasDigit = false;

                    // Greedily consume all valid characters: digits and underscores
                    while (this.pos < this.len && digitOrUscoreRegex.test(this.input[this.pos])) {
                        this.pos++;
                    }

                    // Backtrack if the sequence ends with an underscore.
                    while (this.pos > start && this.input[this.pos - 1] === '_') {
                        this.pos--;
                    }

                    if (this.pos > start) {
                        hasDigit = true;
                    }
                    return hasDigit;
                };

                // 1. Check for Hex Prefix (0x or 0X)
                // Grammar: hexStart(hexDigList)
                if (ch === '0' && this.pos + 1 < this.len && /[xX]/.test(this.input[this.pos + 1])) {
                    this.pos += 2; // Consume '0x'
                    // Attempt to parse hex digits. 
                    if (parseList(true)) {
                        isHex = true;
                    } else {
                        let errCh = this.input[this.pos];
                        throw new Error(`Expected hex digits, but found: ${errCh} at ${this.pos}`);
                    }
                }

                // 2. Parse Integer Part
                // If not hex, parse the initial digits.
                if (!isHex) {
                    parseList(false);
                }

                // 3. Parse Fractional Part (Optional)
                // Grammar: ["." digList] or ["." hexDigList]
                if (this.pos < this.len && this.input[this.pos] === '.') {
                    let nextChar = this.input[this.pos + 1];
                    let digitRegex = isHex ? /[0-9a-fA-F]/ : /[0-9]/;

                    if (nextChar && digitRegex.test(nextChar)) {
                        this.pos++; // Consume '.'
                        parseList(isHex);
                    }
                }

                // 4. Parse Exponent Part (Optional)
                // Grammar: [("e"|"E")[expSign]digList] or [("p"|"P")[expSign]hexDigList]
                if (this.pos < this.len) {
                    let current = this.input[this.pos];
                    let isExp = isHex ? /[pP]/.test(current) : /[eE]/.test(current);

                    if (isExp) {
                        this.pos++; // Consume 'e', 'E', 'p', or 'P'

                        // Optional Sign
                        if (this.pos < this.len && /[+\-]/.test(this.input[this.pos])) {
                            this.pos++;
                        }

                        // Mandatory digits after exponent marker
                        if (parseList(isHex)) {
                            // Success
                        } else {
                            let errCh = this.input[this.pos];
                            throw new Error(`Expected exponent digits, but found: ${errCh} at ${this.pos}`);
                        }
                    }

                    if (this.pos < this.len) {
                        let nextCh = this.input[this.pos];
                        if (/[0-9_a-zA-Z!]/.test(nextCh)) {
                            throw new Error(`Unexpected character: ${nextCh} at ${this.pos}`);
                        }
                    }
                }

                this.tokens.push({ type: 'Numeral', txt: this.input.substring(tokenStart, this.pos), preSpace: preSpace });
                lastNonSpace = this.pos;
                continue;
            }

            // Identifiers / Keywords / Names
            if (/[a-zA-Z_]/.test(ch)) {
                while (this.pos < this.len && /[a-zA-Z0-9_]/.test(this.input[this.pos])) {
                    this.pos++;
                }
                // Macro calls:
                let macro = false;
                if (this.pos < this.len && this.input[this.pos] == "!") {
                    this.pos++;
                    macro = true;
                }
                const txt = this.input.substring(tokenStart, this.pos);
                this.tokens.push({
                    type: (macro || keywords.has(txt) || /^_*$/.test(txt)) ? 'Keyword' : 'Name', // `/^_*$/` => made of only underscores
                    txt: txt,
                    preSpace: preSpace
                });
                lastNonSpace = this.pos;
                continue;
            }

            throw new Error(`Unexpected character: ${ch} at ${this.pos}`);
        }

        // EOF
        let preSpace = "";

        if (lastNonSpace !== this.len) {
            preSpace = this.input.substring(lastNonSpace, this.len);
        }

        this.tokens.push({ type: 'EOF', txt: '', preSpace: preSpace });
    }

    // ========================================================================
    // PARSER UTILS
    // ========================================================================

    peek(offset = 0) {
        return this.tokens[this.tokPos + offset];
    }

    consume() {
        if (this.tokPos < this.tokens.length) {
            return this.tokens[this.tokPos++];
        }
        return { type: 'EOF', txt: '' };
    }

    match(type, txt) {
        const tok = this.peek();
        return tok && tok.type === type && (txt === undefined || tok.txt === txt);
    }

    expect(type, txt) {
        if (this.match(type, txt)) {
            return this.consume();
        }
        const tok = this.peek();
        const msg = `Expected ${type} ${txt ? `"${txt}"` : ''}, found ${tok ? tok.type : 'EOF'} ${tok ? `"${tok.txt}"` : ''} at pos ${this.tokPos}`;
        throw new Error(msg);
    }

    createAstToken(lexerToken) {
        const t = new Token(lexerToken.txt);
        t.preSpace = lexerToken.preSpace;
        return t;
    }

    parseOptToken(txt) {
        const opt = new OptToken(txt);
        if (this.match('Symbol', txt) || this.match('Keyword', txt)) {
            const tok = this.consume();
            opt.present = true;
            opt.txt = tok.txt;
            opt.preSpace = tok.preSpace;
        }
        return opt;
    }

    parseDelimitedList(parserFn, stopTokens) {
        const list = new DelimitedList();
        while (true) {
            const tok = this.peek();
            if (stopTokens.some(t => tok.type === t.type && t.txt === tok.txt)) break;
            if (tok.type === 'EOF') break;

            const item = new DelimitedListItem();
            item.value = parserFn();

            if (this.match('Symbol', ',') || this.match('Symbol', ';')) {
                item.sep = this.createAstToken(this.consume());
            }
            list.items.push(item);
        }
        return list;
    }

    // ========================================================================
    // PARSING IMPLEMENTATION
    // ========================================================================

    parseChunk() {
        const chunk = [];
        while (!this.match('EOF')) {
            chunk.push(this.parseGlobStat());
        }
        return chunk;
    }

    parseModPath() {
        const path = new ModPath();
        if (this.match('Name')) {
            path.root = new ModPathRootName();
            path.root.name = this.parseName();
        } else if (this.match('Keyword', 'self')) {
            path.root = new ModPathRootSelf();
            path.root.kw = this.createAstToken(this.consume());
        } else if (this.match('Keyword', 'crate')) {
            path.root = new ModPathRootCrate();
            path.root.kw = this.createAstToken(this.consume());
        } else if (this.match('Symbol', ':>')) {
            path.root = new ModPathRootOp();
            path.root.op = this.createAstToken(this.consume());
        } else {
            throw new Error("Expected ModPath root");
        }

        while (this.match('Symbol', '::')) {
            const seg = new ModPathSegment();
            seg.sep = this.createAstToken(this.consume());
            seg.name = this.parseName();
            path.segments.push(seg);
        }
        return path;
    }

    parseName() {
        if (this.match('Name')) {
            const n = new Name();
            const tok = this.consume();
            n.name = tok.txt;
            n.preSpace = tok.preSpace;
            return n;
        }
        throw new Error("Expected Name");
    }

    parseStr() {
        if (this.match('LiteralString')) {
            const s = new Str();
            const tok = this.consume();
            s.str = tok.txt;
            s.preSpace = tok.preSpace;
            return s;
        }
        throw new Error("Expected String");
    }

    parseNum() {
        if (this.match('Numeral')) {
            const n = new Num();
            const tok = this.consume();
            n.num = tok.txt;
            n.preSpace = tok.preSpace;
            return n;
        }
        throw new Error("Expected Number");
    }

    // ----------------------------------------------------------------
    // TYPES & PATTERNS
    // ----------------------------------------------------------------

    parseRefAttrs() {
        const attrs = new RefAttrs();
        if (this.match('Keyword', 'in')) {
            attrs.addrspace = { kw: this.createAstToken(this.consume()), name: this.parseName() };
        }
        if (this.match('Symbol', '/')) {
            const names = [];
            do {
                const slash = this.createAstToken(this.consume());
                const n = this.parseName();
                names.push({ kw: slash, l: n });
            } while (this.match('Symbol', '/'));
            attrs.lifetime = names;
        }
        if (this.match('Keyword', 'mut')) {
            attrs.refType = new MutRefType();
            attrs.refType.kw = this.createAstToken(this.consume());
        } else if (this.match('Keyword', 'share')) {
            attrs.refType = new ShareRefType();
            attrs.refType.kw = this.createAstToken(this.consume());
        } else if (this.match('Keyword', 'const')) {
            attrs.refType = new ConstRefType();
            attrs.refType.kw = this.createAstToken(this.consume());
        }
        return attrs;
    }

    parseTypedParam() {
        const p = new TypedParam();
        p.constKw = this.parseOptToken('const');
        p.name = this.parseName();
        p.eq = this.createAstToken(this.expect('Symbol', '='));
        p.type = this.parseExpr();
        return p;
    }

    parseParams() {
        return this.parseDelimitedList(() => this.parseTypedParam(), [{ type: 'Symbol', txt: ')' }]);
    }

    parseTuplableName() {
        if (this.match('Numeral')) {
            const n = new NumTuplableName();
            n.name = this.parseNum();
            return n;
        }
        const n = new NameTuplableName();
        n.name = this.parseName();
        return n;
    }

    parseUncondDestrPat() {
        if (this.match('Symbol', '_')) {
            const pat = new AlwaysDestrPat();
            pat.us = this.createAstToken(this.consume());
            return pat;
        }
        let dspec = null;
        if (!this.match('Symbol', '{'))
            dspec = this.parseExpr(0, true, "spat");

        if (this.match('Symbol', '{')) {

            console.assert(dspec instanceof Expr); // Not OpDestrSpec

            let actualPat = new UncondFieldDestrPat();
            const openBrace = this.createAstToken(this.consume());
            if (this.match('Symbol', '|'))
                actualPat = new UncondPatFieldDestrPat();
            actualPat.openBrace = openBrace;
            actualPat.type = new OptSimplePat();

            if (dspec != null) {
                actualPat.type.present = true;
                actualPat.type.tbl = dspec;
            }

            const items = [];
            while (!this.match('Symbol', '}') && !this.match('EOF')) {
                const li = new DelimitedListItem();
                if (this.match('Symbol', '|')) {
                    if (actualPat instanceof UncondPatFieldDestrPat)
                        throw "TODO";
                    const f = new UncondFieldDestrField();
                    f.openPipe = this.createAstToken(this.consume());
                    f.var = this.parseTuplableName();
                    f.closePipe = this.createAstToken(this.expect('Symbol', '|'));
                    f.pat = this.parseUncondDestrPat();
                    li.value = f;
                } else {
                    if (actualPat instanceof UncondFieldDestrPat)
                        throw "TODO";
                    li.value = this.parseUncondDestrPat();
                }
                if (this.match('Symbol', ',') || this.match('Symbol', ';')) {
                    li.sep = this.createAstToken(this.consume());
                }
                items.push(li);
            }

            const list = new DelimitedList();
            list.items = items;
            actualPat.fields = list;

            actualPat.extraFields = this.parseOptToken('..');//TODO: actually parse this
            actualPat.closeBrace = this.createAstToken(this.expect('Symbol', '}'));
            return actualPat;
        }
        if (!(dspec instanceof OpDestrSpec)) {
            const sp = new SimplePatDestrSpec();
            sp.type = dspec;
            dspec = sp;
        }

        const pat = new UncondVarDestrPat();
        pat.specifiers = dspec;
        pat.name = this.parseName();
        return pat;
    }

    parsePat() {
        if (this.match('Symbol', '_')) {
            const p = new AlwaysDestrPat();
            p.us = this.createAstToken(this.consume());
            return p;
        }

        let dspec = null;
        if (!this.match('Symbol', '{'))
            dspec = this.parseExpr(0, true, "spat");

        if (this.match('Symbol', '{')) {
            console.assert(dspec instanceof Expr); // Not OpDestrSpec

            let actualPat = new FieldDestrPat();
            const openBrace = this.createAstToken(this.consume());
            if (this.match('Symbol', '|'))
                actualPat = new PatFieldDestrPat();
            actualPat.openBrace = openBrace;
            actualPat.type = new OptSimplePat();

            if (dspec != null) {
                actualPat.type.present = true;
                actualPat.type.tbl = dspec;
            }
            const items = [];
            while (!this.match('Symbol', '}') && !this.match('EOF')) {
                const li = new DelimitedListItem();
                if (this.match('Symbol', '|')) {
                    if (actualPat instanceof PatFieldDestrPat)
                        throw "TODO";
                    const f = new FieldDestrField();
                    f.openPipe = this.createAstToken(this.consume());
                    f.var = this.parseTuplableName();
                    f.closePipe = this.createAstToken(this.expect('Symbol', '|'));
                    f.pat = this.parsePat();
                    li.value = f;
                } else {
                    if (actualPat instanceof FieldDestrPat)
                        throw "TODO";
                    li.value = this.parsePat();
                }
                if (this.match('Symbol', ',') || this.match('Symbol', ';')) {
                    li.sep = this.createAstToken(this.consume());
                }
                items.push(li);
            }
            const list = new DelimitedList();
            list.items = items;
            actualPat.fields = list;

            actualPat.extraFields = this.parseOptToken('..');//TODO: actually parse this
            actualPat.closeBrace = this.createAstToken(this.expect('Symbol', '}'));
            return actualPat;
        }
        if (!(dspec instanceof OpDestrSpec)) {
            const sp = new SimplePatDestrSpec();
            sp.type = dspec;
            dspec = sp;
        }
        const name = this.parseName(); // TODO: expr only! (check for =>)

        if (this.match('Symbol', '=')) {
            const eq = this.createAstToken(this.consume());
            const val = this.parseExpr();

            const pat = new VarDestrPat();
            pat.base = new UncondVarDestrPat();
            pat.base.specifiers = dspec;
            pat.base.name = name;
            pat.eq = eq;
            pat.valPat = new SimplePat();
            pat.valPat.expr = val;
            return pat;
        }
        const pat = new UncondVarDestrPat();
        pat.specifiers = dspec;
        pat.name = name;
        return pat;
    }

    // ----------------------------------------------------------------
    // EXPRESSIONS
    // ----------------------------------------------------------------

    parsePrimary(specType = "") {
        if (this.match('Symbol', '(')) {
            const e = new ParenExpr();
            e.openParen = this.createAstToken(this.consume());
            e.expr = this.parseExpr();
            e.closeParen = this.createAstToken(this.expect('Symbol', ')'));
            return e;
        }

        if (this.match('LiteralString')) {
            const e = new StrExpr();
            e.raw = this.parseStr();
            return e;
        }

        if (this.match('Numeral')) {
            const e = new NumExpr();
            e.raw = this.parseNum();
            return e;
        }

        if (this.match('Name') || this.match('Keyword', 'self') || this.match('Keyword', 'crate') || this.match('Symbol', ':>')) {
            if (specType == "spat" && this.match('Name')) {
                let afterTok = this.peek(1);
                let spp = afterTok.type == 'Symbol' && (afterTok.txt == '=' || afterTok.txt == '=>' || afterTok.txt == ',' || afterTok.txt == '}');
                spp |= afterTok.type == 'Keyword' && (afterTok.txt == 'in');
                spp |= afterTok.type == 'EOF';

                if (spp)
                    return new OpDestrSpec();
            }
            const e = new ModPathExpr();
            e.path = this.parseModPath();
            return e;
        }

        if (specType != "spat" && this.match('Symbol', '{')) {
            return this.parseTableConstructor();
        }

        if (this.match('Symbol', "'")) {
            const labelStart = this.createAstToken(this.consume());
            const label = this.parseName();
            const colon = this.createAstToken(this.expect('Symbol', ':'));

            if (this.match('Keyword', 'loop')) return this.parseLoopExpr(labelStart, label, colon);
            if (this.match('Keyword', 'do')) return this.parseDoExpr(labelStart, label, colon);
        }

        if (this.match('Keyword', 'loop')) return this.parseLoopExpr(new OptToken(), new Name(), new Token(":"));

        if (this.match('Keyword', 'match')) {
            const e = new MatchExpr();
            e.matchKw = this.createAstToken(this.consume());
            e.argument = this.parseExpr(0, true);
            e.block = this.parseMatchTypeBlock();
            return e;
        }

        if (this.match('Keyword', 'TODO!')) {
            const e = new TodoExpr();
            e.kw = this.createAstToken(this.consume());
            e.msg = this.parseStr();
            return e;
        }

        if (this.match('Keyword', '_COMP_TODO!')) {
            const e = new CompTodoExpr();
            e.kw = this.createAstToken(this.consume());
            e.openParen = this.createAstToken(this.expect('Symbol', '('));
            e.msg = this.parseStr();
            e.comma = this.createAstToken(this.expect('Symbol', ','));
            e.expr = this.parseExpr();
            e.closeParen = this.createAstToken(this.expect('Symbol', ')'));
            return e;
        }

        if (this.match('Symbol', '_')) {
            const e = new UnderscoreExpr();
            e.us = this.createAstToken(this.consume());
            return e;
        }

        if (this.match('Keyword', 'const')) {
            const kw = this.consume();
            if (this.match('Symbol', '(')) {
                const e = new ConstExpr();
                e.constKw = this.createAstToken(kw);
                e.openParen = this.createAstToken(this.consume());
                e.expr = this.parseExpr();
                e.closeParen = this.createAstToken(this.expect('Symbol', ')'));
                return e;
            }
            throw new Error("Expected '(' after const in expression");
        }

        if (specType != "spat" && (this.match('Keyword', 'safe') || this.match('Keyword', 'unsafe') || this.match('Symbol', '|'))) {
            return this.parseLambdaExpr();
        }

        if (this.match('Keyword', 'do')) {
            return this.parseDoExpr(new OptToken(), new Name(), new Token(":"));
        }

        throw new Error("Unexpected token in expression: " + this.peek().txt);
    }

    parseLoopExpr(labelStart, label, colon) {
        const e = new LoopExpr();
        e.labelStart = labelStart;
        e.label = label;
        e.labelColon = colon;
        e.loopKw = this.createAstToken(this.expect('Keyword', 'loop'));
        e.retArrow = this.parseOptToken('->');
        if (e.retArrow.present) e.retType = this.parseExpr(0, true);
        e.body = this.parseBlock();
        return e;
    }

    parseDoExpr(labelStart, label, colon) {
        const e = new DoExpr();
        e.labelStart = labelStart;
        e.label = label;
        e.labelColon = colon;
        e.constKw = this.parseOptToken('const');
        e.doKw = this.createAstToken(this.expect('Keyword', 'do'));
        e.retArrow = this.parseOptToken('->');
        if (e.retArrow.present) e.retType = this.parseExpr(0, true);
        e.block = this.parseBlock();
        return e;
    }

    parseLambdaExpr() {
        const e = new LambdaExpr();
        e.safety = new Safety();
        if (this.match('Keyword', 'safe')) { e.safety.kind = "safe"; e.safety.kw = this.createAstToken(this.consume()); }
        if (this.match('Keyword', 'unsafe')) { e.safety.kind = "unsafe"; e.safety.kw = this.createAstToken(this.consume()); }

        e.pipe1 = this.createAstToken(this.expect('Symbol', '|'));
        e.params = this.parseParams();
        e.pipe2 = this.createAstToken(this.expect('Symbol', '|'));

        e.retArrow = this.parseOptToken('->');
        if (e.retArrow.present) {
            e.retType = this.parseExpr();
            e.doubleArrow = this.createAstToken(this.expect('Symbol', '=>'));
        }

        e.body = this.parseExpr();
        return e;
    }

    parseUnary(basic = false, specType = "") {
        const ops = [];
        while (true) {
            if (this.match('Symbol', '-')) ops.push({ type: 'UnOp', op: this.createAstToken(this.consume()) });
            else if (this.match('Symbol', '!')) ops.push({ type: 'UnOp', op: this.createAstToken(this.consume()) });
            else if (this.match('Symbol', '~')) ops.push({ type: 'UnOp', op: this.createAstToken(this.consume()) });
            else if (this.match('Symbol', 'ex')) ops.push({ type: 'UnOp', op: this.createAstToken(this.consume()) });
            else if (this.match('Symbol', '*')) {
                const op = new RefTypePreOp(); op.star = this.createAstToken(this.consume()); op.attrs = this.parseRefAttrs(); ops.push(op);
            }
            else if (this.match('Symbol', '&')) {
                const op = new RefPreOp(); op.amp = this.createAstToken(this.consume()); op.attrs = this.parseRefAttrs(); ops.push(op);
            }
            else if (this.match('Symbol', '[')) {
                const op = new SlicePreOp(); op.open = this.createAstToken(this.consume()); op.close = this.createAstToken(this.expect('Symbol', ']')); ops.push(op);
            }
            else if (this.match('Keyword', 'dyn')) { const op = new DynPreOp(); op.kw = this.createAstToken(this.consume()); ops.push(op); }
            else if (this.match('Keyword', 'impl')) { const op = new ImplPreOp(); op.kw = this.createAstToken(this.consume()); ops.push(op); }
            else if (this.match('Keyword', 'union')) { const op = new UnionPreOp(); op.kw = this.createAstToken(this.consume()); ops.push(op); }
            else if (this.match('Keyword', 'mut')) { const op = new MutPreOp(); op.kw = this.createAstToken(this.consume()); ops.push(op); }
            else if (this.match('Symbol', '..')) { const op = new RangePreOp(); op.op = this.createAstToken(this.consume()); ops.push(op); }
            else if (this.match('Symbol', '@')) { ops.push(new AnnotationPreOp(this.parseAnnotation())); }
            else if (this.match('Keyword', 'if')) {
                const op = new IfPreOp();
                op.ifKw = this.createAstToken(this.consume());
                op.expr = this.parseExpr();
                op.arrow = this.createAstToken(this.expect('Symbol', '=>'));
                ops.push(op);
            }
            else break;
        }

        let primary = this.parsePrimary(specType);

        if (specType == "spat" && primary instanceof OpDestrSpec) {
            let first = true;
            for (let i in ops) {
                let v = ops[i];
                if (first && v instanceof MutPreOp) {
                    primary.mutKw = v.kw;
                }
                else if (v instanceof RefTypePreOp) {
                    primary.ops.push(v);
                } else {
                    throw new Error(`Unexpected operator (${v.type}) for pattern at pos ${this.tokPos}`);
                }
                first = false;
            }
            return primary;
        }
        //TODO: better postfix range handling! `1.. return` is a post op, `1..1` is a bin op.

        const sufOps = [];
        while (true) {
            if (this.match('Symbol', '.*')) {
                const op = new DerefSubVar(); op.op = this.createAstToken(this.consume()); sufOps.push(op);
            }
            else if (this.match('Symbol', '.')) {
                const dot = this.consume();
                if (this.match('Symbol', ':')) {
                    const op = new ConstDotSubVar(); op.op = this.createAstToken(dot); op.field = this.parseName(); sufOps.push(op);
                } else if (this.match('Symbol', '(') || this.match('LiteralString') || this.match('Numeral')) {
                    const call = new SelfableCall();
                    call.dot = this.createAstToken(dot);
                    call.method = this.parseName();
                    call.args = this.parseArgs();
                    sufOps.push(call);
                } else {
                    const op = new DotSubVar(); op.op = this.createAstToken(dot); op.field = this.parseTuplableName(); sufOps.push(op);
                }
            }
            else if (this.match('Symbol', ':')) {
                if (this.match('Symbol', '[')) {
                    const op = new IdxSubVar();
                    op.open = this.createAstToken(this.consume());
                    op.expr = this.parseExpr();
                    op.close = this.createAstToken(this.expect('Symbol', ']'));
                    sufOps.push(op);
                } else break;
            }
            else if (this.match('Symbol', '?')) { const op = new SufOp(); op.op = this.createAstToken(this.consume()); sufOps.push(op); }
            else if (this.match('Symbol', '..')) { const op = new SufOp(); op.op = this.createAstToken(this.consume()); sufOps.push(op); }
            else if (this.match('Keyword', 'try')) {
                const op = new TryOp(); op.tryKw = this.createAstToken(this.consume()); op.block = this.parseMatchTypeBlock(); sufOps.push(op);
            }
            else if (this.match('Symbol', '(') || this.match('LiteralString') || this.match('Numeral') || (this.match('Symbol', '{') && !basic)) {
                const call = new SelfableCall();
                call.args = this.parseArgs();
                sufOps.push(call);
            }
            else break;
        }

        if (ops.length > 0 || sufOps.length > 0) {
            const u = new UnaryExpr();
            u.PreOps = ops;
            u.primary = primary;
            u.sufOps = sufOps;
            return u;
        }
        return primary;
    }

    parseArgs() {
        if (this.match('Symbol', '(')) {
            const a = new ParenArgs();
            a.openParen = this.createAstToken(this.consume());
            a.args = this.parseParams();
            a.closeParen = this.createAstToken(this.expect('Symbol', ')'));
            return a;
        }
        if (this.match('LiteralString')) {
            const a = new StrArgs();
            a.args = this.parseStr();
            return a;
        }
        if (this.match('Numeral')) {
            const a = new NumArgs();
            a.args = this.parseNum();
            return a;
        }
        if (this.match('Symbol', '{')) {
            const a = new TableArgs();
            a.args = this.parseTableConstructor();
            return a;
        }
        throw new Error("Expected arguments");
    }

    parseExpr(precedence = 0, basic = false, specType = "") {
        if (this.match('Keyword', 'fn')) return this.parseFnExpr();

        let left = this.parseUnary(basic, specType);

        while (true) {
            const opTok = this.peek();
            if (this.isBinOp(opTok, specType)) {
                const p = this.getPrecedence(opTok.txt);
                if (p >= precedence) {
                    this.consume();
                    const right = this.parseExpr(p + 1, basic, specType);
                    const bin = new BinExpr();
                    bin.left = left;
                    bin.op = this.createAstToken(opTok);
                    bin.right = right;
                    left = bin;
                    continue;
                }
            }
            break;
        }
        return left;
    }

    parseFnExpr() {
        const e = new FnExpr();
        e.safety = new Safety();
        if (this.match('Keyword', 'safe')) { e.safety.kind = "safe"; e.safety.kw = this.createAstToken(this.consume()); }
        if (this.match('Keyword', 'unsafe')) { e.safety.kind = "unsafe"; e.safety.kw = this.createAstToken(this.consume()); }

        e.fnKw = this.createAstToken(this.expect('Keyword', 'fn'));
        e.openParen = this.createAstToken(this.expect('Symbol', '('));

        if (this.match('Symbol', '*')) {
            const star = this.createAstToken(this.consume());
            const attrs = this.parseRefAttrs();
            e.selfParamRefs.push(new RefTypePreOp(star, attrs));
        }

        if (this.match('Keyword', 'self')) {
            e.selfParam = new OptToken("self");
            e.selfParam.present = true;
            e.selfParam.txt = this.consume().txt;
            e.selfParamDelim = this.createAstToken(this.consume());
            if (!this.match('Symbol', ')')) e.params = this.parseParams();
        } else {
            e.params = this.parseParams();
        }

        e.closeParen = this.createAstToken(this.expect('Symbol', ')'));

        e.retArrow = this.parseOptToken('->');
        if (e.retArrow.present) e.retType = this.parseExpr(0, true);

        e.body = new OptBlockNode();
        if (this.match('Symbol', '{')) {
            e.body.present = true;
            e.body.block = this.parseBlock();
        }

        return e;
    }

    isBinOp(tok, specType = "") {
        if (!tok) return false;
        const ops = ["+", "-", "*", "/", "//", "^", "%", "++", "<", "<=", ">", ">=", "==", "!=", "~", "|", "..", "else", "**", "as"];
        if (specType != "spat") {
            ops.push("and", "or");
        }
        return ops.includes(tok.txt);
    }

    getPrecedence(op) {
        const p = {
            "or": 1, "and": 2, "else": 3,
            "==": 4, "!=": 4, "<": 4, "<=": 4, ">": 4, ">=": 4,
            "|": 5, "~": 6, "..": 7,
            "**": 8, "++": 9,
            "*": 10, "/": 10, "//": 10, "%": 10,
            "+": 10, "-": 10,
            "as": 11
        };
        return p[op] || 0;
    }

    // ----------------------------------------------------------------
    // STATEMENTS & BLOCKS
    // ----------------------------------------------------------------

    parseBlock() {
        const b = new BlockNode();
        b.openBrace = this.createAstToken(this.expect('Symbol', '{'));

        while (!this.match('Symbol', '}')) {
            if (this.match('EOF')) break;
            const tok = this.peek();
            if (this.match('Keyword', 'return') || this.match('Keyword', 'break') || this.match('Keyword', 'continue') || this.match('Keyword', 'throw')) {
                b.retStat = this.parseRetStat();
                if (this.match('Symbol', ';'))
                    b.semicol = this.createAstToken(this.consume());
                break;
            }
            b.stats.push(this.parseStat());
        }

        b.closeBrace = this.createAstToken(this.expect('Symbol', '}'));
        return b;
    }

    parseRetStat() {
        const tok = this.peek();
        if (this.match('Keyword', 'return')) {
            const s = new ReturnStat();
            s.returnKw = this.createAstToken(this.consume());
            s.expr = new OptExpr();
            if (!this.match('Symbol', ';') && !this.match('Symbol', '}')) {
                s.expr.present = true;
                s.expr.expr = this.parseExpr();
            }
            return s;
        }
        if (this.match('Keyword', 'break')) {
            const s = new BreakStat();
            s.breakKw = this.createAstToken(this.consume());
            s.label = this.parseOptToken("'");
            if (s.label.present) s.labelName = this.parseName();
            s.expr = new OptExpr();
            if (!this.match('Symbol', ';') && !this.match('Symbol', '}')) {
                s.expr.present = true;
                s.expr.expr = this.parseExpr();
            }
            return s;
        }
        if (this.match('Keyword', 'continue')) {
            const s = new ContinueStat();
            s.continueKw = this.createAstToken(this.consume());
            s.label = this.parseOptToken("'");
            if (s.label.present) s.labelName = this.parseName();
            s.expr = new OptExpr();
            if (!this.match('Symbol', ';') && !this.match('Symbol', '}')) {
                s.expr.present = true;
                s.expr.expr = this.parseExpr();
            }
            return s;
        }
        if (this.match('Keyword', 'throw')) {
            const s = new ThrowStat();
            s.throwKw = this.createAstToken(this.consume());
            s.expr = this.parseExpr();
            return s;
        }
        throw new Error("Expected return statement");
    }

    parseStat() {
        if (this.match('Symbol', ';')) {
            const s = new EmptyStat();
            s.semicol = this.createAstToken(this.consume());
            return s;
        }

        const anns = this.parseAnnotations();
        if (anns.length > 0) {
            const inner = this.parseStat();
            return inner;
        }

        if (this.match('Keyword', 'let')) {
            const s = new LetStat();
            s.letKw = this.createAstToken(this.consume());
            s.pattern = this.parseUncondDestrPat();
            s.eq = this.parseOptToken('=');
            if (s.eq.present) s.value = this.parseExpr();
            return s;
        }

        if (this.match('Keyword', 'drop')) {
            const s = new DropStat();
            s.dropKw = this.createAstToken(this.consume());
            s.expr = this.parseExpr();
            return s;
        }

        if (this.match('Keyword', 'if')) return this.parseIfStat();

        if (this.match('Keyword', 'unsafe')) {
            const s = new UnsafeStat();
            s.unsafeKw = this.createAstToken(this.consume());
            s.openBrace = this.createAstToken(this.expect('Symbol', '{'));
            while (!this.match('Symbol', '}')) s.stats.push(this.parseStat());
            s.closeBrace = this.createAstToken(this.consume());
            return s;
        }

        if (this.match('Symbol', "'")) {
            const labelStart = this.createAstToken(this.consume());
            const label = this.parseName();
            const colon = this.createAstToken(this.expect('Symbol', ':'));
            if (this.match('Keyword', 'while')) return this.parseWhileStat(labelStart, label, colon);
            if (this.match('Keyword', 'for')) return this.parseForStat(labelStart, label, colon);
            throw new Error("Expected while/for after label");
        }

        if (this.match('Keyword', 'while')) return this.parseWhileStat(new OptToken(), new Name(), new Token(":"));
        if (this.match('Keyword', 'for')) return this.parseForStat(new OptToken(), new Name(), new Token(":"));

        if (this.match('Keyword', 'struct') || this.match('Keyword', 'enum') || this.match('Keyword', 'fn') || this.match('Keyword', 'trait') || this.match('Keyword', 'extern') || this.match('Keyword', 'impl') || this.match('Keyword', 'use') || this.match('Keyword', 'mod') || this.match('Keyword', 'const') || this.match('Keyword', 'union') || this.match('Keyword', 'ex')) {
            return this.parseGlobStat();
        }

        if (this.match('Name') || this.match('Symbol', '(')) {
            const v = this.parseVar();
            if (this.match('Symbol', '=')) {
                const s = new AssignStat();
                s.var = v;
                s.eq = this.createAstToken(this.consume());
                s.expr = this.parseExpr();
                return s;
            }
            if (this.match('Symbol', '(') || this.match('LiteralString') || this.match('Numeral') || this.match('Symbol', '{')) {
                const s = new CallStat();
                s.var = v;
                s.call = new SelfableCall();
                s.call = this.parseArgs();
                return s;
            }
            return v;
        }

        let isTd = this.match('Keyword', 'TODO!');
        if (isTd || this.match('Keyword', 'loop') || this.match('Keyword', 'match') || isTd) {
            const e = this.parseExpr(0, !isTd);
            if (e instanceof LoopExpr) {
                const s = new LoopStat();
                s.labelStart = e.labelStart; s.label = e.label; s.labelColon = e.labelColon;
                s.loopKw = e.loopKw; s.retArrow = e.retArrow; s.retType = e.retType; s.body = e.body;
                return s;
            }
            return e;
        }

        throw new Error("Unknown statement");
    }

    parseIfStat() {
        const s = new IfStat();
        s.ifKw = this.createAstToken(this.consume());
        s.condition = this.parseExpr(0, true);
        s.consequent = this.parseBlockOrRet();

        while (this.match('Keyword', 'else')) {
            const elseKw = this.createAstToken(this.consume());
            if (this.match('Keyword', 'if')) {
                const alt = new IfAlternate();
                alt.elseKw = elseKw;
                alt.ifKw = this.createAstToken(this.consume());
                alt.condition = this.parseExpr(0, true);
                alt.body = this.parseBlockOrRet();
                s.alternates.push(alt);
            } else {
                s.elseKw = new OptToken("else");
                s.elseKw.present = true;
                s.elseKw.txt = "else";
                s.elseBlock = this.parseBlockOrRet();
                break;
            }
        }
        return s;
    }

    parseBlockOrRet() {
        if (this.match('Symbol', '{')) {
            const b = new BlockOrRetBlock();
            b.block = this.parseBlock();
            return b;
        }
        const r = new BlockOrRetRetStat();
        r.retStat = this.parseRetStat();
        if (this.match('Symbol', ';')) r.semicol = this.createAstToken(this.consume());
        return r;
    }

    parseWhileStat(labelStart, label, colon) {
        const s = new WhileStat();
        s.labelStart = labelStart; s.label = label; s.labelColon = colon;
        s.whileKw = this.createAstToken(this.expect('Keyword', 'while'));
        s.condition = this.parseExpr(0, true);
        s.body = this.parseBlock();
        return s;
    }

    parseForStat(labelStart, label, colon) {
        const s = new ForStat();
        s.labelStart = labelStart; s.label = label; s.labelColon = colon;
        s.forKw = this.createAstToken(this.expect('Keyword', 'for'));
        s.constKw = this.parseOptToken('const');
        s.pattern = this.parseUncondDestrPat();
        s.inKw = this.createAstToken(this.expect('Keyword', 'in'));
        s.iterable = this.parseExpr(0, true);
        s.body = this.parseBlock();
        return s;
    }

    parseVar() {
        const v = new Var();
        if (this.match('Name')) v.root = this.parseName();
        else if (this.match('Symbol', '(')) {
            this.consume();
            v.root = this.parseExpr();
            this.expect('Symbol', ')');
        }

        while (true) {
            if (this.match('Symbol', '.*')) {
                v.suffixes.push(new DerefSubVar(this.createAstToken(this.consume())));
            } else if (this.match('Symbol', '.')) {
                const dot = this.consume();
                if (this.match('Symbol', ':')) {
                    v.suffixes.push(new ConstDotSubVar(this.createAstToken(dot), this.parseName()));
                } else if (this.match('Symbol', '(') || this.match('LiteralString') || this.match('Numeral')) {
                    const call = new SelfableCall();
                    call.dot = this.createAstToken(dot);
                    call.method = this.parseName();
                    call.args = this.parseArgs();
                    v.suffixes.push(call);
                } else {
                    v.suffixes.push(new DotSubVar(this.createAstToken(dot), this.parseTuplableName()));
                }
            } else if (this.match('Symbol', ':')) {
                if (this.match('Symbol', '[')) {
                    const op = new IdxSubVar();
                    op.open = this.createAstToken(this.consume());
                    op.expr = this.parseExpr();
                    op.close = this.createAstToken(this.expect('Symbol', ']'));
                    v.suffixes.push(op);
                } else break;
            } else if (this.match('Symbol', '?')) { v.suffixes.push(new SufOp(this.createAstToken(this.consume()))); }
            else if (this.match('Symbol', '..')) { v.suffixes.push(new SufOp(this.createAstToken(this.consume()))); }
            else if (this.match('Keyword', 'try')) {
                v.suffixes.push(new TryOp(this.createAstToken(this.consume()), this.parseMatchTypeBlock()));
            }
            else if (this.match('Symbol', '(') || this.match('LiteralString') || this.match('Numeral') || this.match('Symbol', '{')) {
                v.suffixes.push(new SelfableCall(null, null, this.parseArgs()));
            } else break;
        }
        return v;
    }

    // ----------------------------------------------------------------
    // GLOBAL DECLARATIONS
    // ----------------------------------------------------------------

    parseGlobStat() {
        if (this.match('Symbol', ';')) return new EmptyGlobStat(this.createAstToken(this.consume()));

        const exportKw = new Export();
        if (this.match('Keyword', 'ex')) {
            exportKw.kw.present = true;
            exportKw.kw.txt = this.consume().txt;
        }

        if (this.match('Keyword', 'struct')) {
            const s = new StructDecl();
            s.export = exportKw;
            s.structKw = this.createAstToken(this.consume());
            s.name = this.parseName();
            if (this.match('Symbol', '(')) {
                s.openParen = this.createAstToken(this.consume());
                s.params = this.parseParams();
                s.closeParen = this.createAstToken(this.expect('Symbol', ')'));
            }
            s.body = this.parseTableConstructor();
            return s;
        }

        if (this.match('Keyword', 'enum')) {
            const s = new EnumDecl();
            s.export = exportKw;
            s.enumKw = this.createAstToken(this.consume());
            s.name = this.parseName();
            if (this.match('Symbol', '(')) {
                s.openParen = this.createAstToken(this.consume());
                s.params = this.parseParams();
                s.closeParen = this.createAstToken(this.expect('Symbol', ')'));
            }
            s.openBrace = this.createAstToken(this.expect('Symbol', '{'));
            s.fields = this.parseDelimitedList(() => this.parseEnumField(), [{ type: 'Symbol', txt: '}' }]);
            if (this.match('Symbol', '..')) s.spread = this.createAstToken(this.consume());
            s.closeBrace = this.createAstToken(this.expect('Symbol', '}'));
            return s;
        }

        if (this.match('Keyword', 'fn')) {
            const s = new FunctionDecl();
            s.export = exportKw;
            if (this.match('Keyword', 'safe')) { s.safety.kind = "safe"; s.safety.kw = this.createAstToken(this.consume()); }
            if (this.match('Keyword', 'unsafe')) { s.safety.kind = "unsafe"; s.safety.kw = this.createAstToken(this.consume()); }
            s.structKw = this.parseOptToken('struct');
            s.fnKw = this.createAstToken(this.consume());
            s.name = this.parseName();
            s.openParen = this.createAstToken(this.expect('Symbol', '('));
            s.params = this.parseParams();
            s.closeParen = this.createAstToken(this.expect('Symbol', ')'));
            s.retArrow = this.parseOptToken('->');
            if (s.retArrow.present) s.retType = this.parseExpr(0, true);
            s.body = new OptBlockNode();
            if (this.match('Symbol', '{')) {
                s.body.present = true;
                s.body.block = this.parseBlock();
            }
            return s;
        }

        if (this.match('Keyword', 'trait')) {
            const s = new TraitDecl();
            s.export = exportKw;
            s.traitKw = this.createAstToken(this.consume());
            s.name = this.parseName();
            if (this.match('Symbol', '(')) {
                s.openParen = this.createAstToken(this.consume());
                s.params = this.parseParams();
                s.closeParen = this.createAstToken(this.expect('Symbol', ')'));
            }
            if (this.match('Keyword', 'where')) s.where = this.parseWhereClauses();
            s.body = this.parseTableConstructor();
            return s;
        }

        if (this.match('Keyword', 'extern')) {
            const s = new ExternBlock();
            if (this.match('Keyword', 'safe')) { s.safety.kind = "safe"; s.safety.kw = this.createAstToken(this.consume()); }
            if (this.match('Keyword', 'unsafe')) { s.safety.kind = "unsafe"; s.safety.kw = this.createAstToken(this.consume()); }
            s.externKw = this.createAstToken(this.consume());
            s.abiName = this.parseStr();
            s.openBrace = this.createAstToken(this.expect('Symbol', '{'));
            while (!this.match('Symbol', '}')) s.stats.push(this.parseGlobStat());
            s.closeBrace = this.createAstToken(this.consume());
            return s;
        }

        if (this.match('Keyword', 'safe') || this.match('Keyword', 'unsafe') || this.match('Keyword', 'impl')) {
            const s = new ImplDecl();
            s.export = exportKw;
            if (this.match('Keyword', 'safe')) { s.safety.kind = "safe"; s.safety.kw = this.createAstToken(this.consume()); }
            if (this.match('Keyword', 'unsafe')) { s.safety.kind = "unsafe"; s.safety.kw = this.createAstToken(this.consume()); }
            s.implKw = this.createAstToken(this.expect('Keyword', 'impl'));
            if (this.match('Symbol', '(')) {
                s.openParen = this.createAstToken(this.consume());
                s.params = this.parseParams();
                s.closeParen = this.createAstToken(this.expect('Symbol', ')'));
            }
            const maybeTrait = this.parseExpr(0, true);
            if (this.match('Keyword', 'for')) {
                s.forKw = this.createAstToken(this.consume());
                s.traitType = maybeTrait;
                s.targetType = this.parseExpr(0, true);
            } else {
                s.targetType = maybeTrait;
            }
            if (this.match('Keyword', 'where')) s.where = this.parseWhereClauses();
            s.body = this.parseTableConstructor();
            return s;
        }

        if (this.match('Keyword', 'use')) {
            const s = new UseDecl();
            s.export = exportKw;
            s.useKw = this.createAstToken(this.consume());
            s.path = this.parseModPath();
            s.variant = this.parseUseVariant();
            return s;
        }

        if (this.match('Keyword', 'mod')) {
            const s = new ModDecl();
            s.export = exportKw;
            s.modKw = this.createAstToken(this.consume());
            s.name = this.parseName();
            if (this.match('Symbol', '{')) {
                s.openBrace = this.createAstToken(this.consume());
                while (!this.match('Symbol', '}')) s.chunk.push(this.parseGlobStat());
                s.closeBrace = this.createAstToken(this.consume());
            }
            return s;
        }

        if (this.match('Keyword', 'const')) {
            const s = new ConstDecl();
            s.export = exportKw;
            s.constKw = this.createAstToken(this.consume());
            s.pattern = this.parseUncondDestrPat();
            s.eq = this.createAstToken(this.expect('Symbol', '='));
            s.value = this.parseExpr();
            return s;
        }

        if (this.match('Keyword', 'union')) {
            const s = new UnionDecl();
            s.export = exportKw;
            s.unionKw = this.createAstToken(this.consume());
            s.name = this.parseName();
            if (this.match('Symbol', '(')) {
                s.openParen = this.createAstToken(this.consume());
                s.params = this.parseParams();
                s.closeParen = this.createAstToken(this.expect('Symbol', ')'));
            }
            s.body = this.parseTableConstructor();
            return s;
        }

        throw new Error("Unknown global declaration");
    }

    parseEnumField() {
        const f = new EnumField();
        f.annotations = this.parseAnnotations();
        if (this.match('Keyword', 'ex')) {
            f.export.kw.present = true;
            f.export.kw.txt = this.consume().txt;
        }
        f.name = this.parseName();
        if (this.match('Symbol', '{')) {
            f.table.present = true;
            f.table.tbl = this.parseTableConstructor();
        }
        f.outerAnnotations = this.parseAnnotations(true);
        return f;
    }

    parseWhereClauses() {
        const w = new WhereClauses();
        w.whereKw = this.createAstToken(this.expect('Keyword', 'where'));
        w.clauses = this.parseDelimitedList(() => this.parseWhereClause(), [{ type: 'Symbol', txt: '{' }]);
        return w;
    }

    parseWhereClause() {
        const c = new WhereClause();
        if (this.match('Keyword', 'Self')) {
            c.name = new Name(); c.name.name = this.consume().txt;
        } else {
            c.name = this.parseName();
        }
        c.colon = this.createAstToken(this.expect('Symbol', ':'));
        c.type = this.parseExpr(0, true);
        return c;
    }

    parseUseVariant() {
        if (this.match('Keyword', 'as')) {
            const v = new UseAs();
            v.asKw = this.createAstToken(this.consume());
            v.name = this.parseName();
            return v;
        }
        if (this.match('Symbol', '::*')) {
            const v = new StarUseVariant();
            v.kw = this.createAstToken(this.consume());
            return v;
        }
        if (this.match('Symbol', '::')) {
            const cc = this.peek(1);
            if (cc && cc.txt === '{') {
                const v = new BraceUseVariant();
                v.colonColon = this.createAstToken(this.consume());
                v.openBrace = this.createAstToken(this.consume());
                if (this.match('Keyword', 'self')) {
                    v.selfKw.present = true;
                    v.selfKw.txt = this.consume().txt;
                    if (this.match('Symbol', ',') || this.match('Symbol', ';')) v.selfDelim = this.createAstToken(this.consume());
                }
                v.items = this.parseDelimitedList(() => this.parseName(), [{ type: 'Symbol', txt: '}' }]);
                v.closeBrace = this.createAstToken(this.expect('Symbol', '}'));
                return v;
            }
        }
        return new SimpleUseVariant();
    }

    parseTableConstructor() {
        const t = new TableConstructor();
        t.openBrace = this.createAstToken(this.expect('Symbol', '{'));

        while (!this.match('Symbol', '}')) {
            const li = new DelimitedListItem();
            if (this.match('Name')) {
                const peekEq = this.peek(1);
                if (peekEq && peekEq.txt === '=') {
                    const f = new NamedField();
                    f.name = this.parseName();
                    f.eq = this.createAstToken(this.consume());
                    f.expr = this.parseExpr();
                    li.value = f;
                    if (this.match('Symbol', ',') || this.match('Symbol', ';')) li.sep = this.createAstToken(this.consume());
                    t.fields.items.push(li);
                    continue;
                }
            }

            const f = new ExprField();
            f.expr = this.parseExpr();
            li.value = f;
            if (this.match('Symbol', ',') || this.match('Symbol', ';')) li.sep = this.createAstToken(this.consume());
            t.fields.items.push(li);
        }

        t.closeBrace = this.createAstToken(this.consume());
        return t;
    }

    parseMatchTypeBlock() {
        const b = new MatchTypeBlock();
        if (this.match('Symbol', '->')) {
            b.retArrow = this.createAstToken(this.consume());
            b.retType = this.parseExpr(0, true);
        }
        b.openBrace = this.createAstToken(this.expect('Symbol', '{'));
        b.items = this.parseDelimitedList(() => this.parseMatchItem(), [{ type: 'Symbol', txt: '}' }]);
        b.closeBrace = this.createAstToken(this.consume());
        return b;
    }

    parseMatchItem() {
        const i = new MatchItem();
        i.pat = this.parsePat();
        if (this.match('Keyword', 'if')) {
            i.ifKw.present = true;
            i.ifKw.txt = this.consume().txt;
            i.ifExpr = this.parseExpr();
        }
        i.arrow = this.createAstToken(this.expect('Symbol', '=>'));
        i.expr = this.parseExpr(0, true);
        return i;
    }

    parseAnnotations(isOuter = false) {
        const annotations = [];
        while (true) {
            if (this.match('Symbol', isOuter ? '@<' : '@')) {
                const ann = new (isOuter ? OuterModPathAnnotation : ModPathAnnotation)();
                ann.at = this.createAstToken(this.consume());
                ann.path = this.parseModPath();
                if (this.match('Symbol', '{')) {
                    ann.table.present = true;
                    ann.table.tbl = this.parseTableConstructor();
                }
                annotations.push(ann);
            }
            else if (this.match('Symbol', isOuter ? '--<' : '---')) {
                const startTok = this.consume();
                if (this.match('LiteralString')) {
                    const strTok = this.consume();
                    const ann = new (isOuter ? OuterDocLineAnnotation : DocLineAnnotation)();
                    ann.txt = this.createAstToken(startTok);
                    ann.content = strTok.txt;
                    annotations.push(ann);
                } else if (this.match('LineOfText')) {
                    const txtTok = this.consume();
                    const ann = new (isOuter ? OuterDocLineAnnotation : DocLineAnnotation)();
                    ann.txt = this.createAstToken(startTok);
                    ann.content = txtTok.txt;
                    annotations.push(ann);
                } else {
                    throw new Error("Expected string or text after comment annotation");
                }
            } else {
                break;
            }
        }
        return annotations;
    }
}
