class Preprocessor:
    """
    字符级的预处理类
    """

    def __init__(self, exclude_from_norm=None):
        """Initializes the tokenizer.

        Args:
            exclude_from_norm (optional) - list of chars to exclude from normalization
        """

        self.PLACESYM = set([u'{', u'}', u'[', u']', u'-', u'_'])
        self.FACESYM = set([u'{', u'}', u'[', u']', u'_', u'^', u'=', u'(', u')', u'→', u'-', u'﹏', u'●', u'❁', u'⊙', \
                       u'▿', u'˙', u'▽', u'￣', u'٩', u'๑', u'‾', u'⍨', u'—', u'_', u'<', u'>', u'╯', u'╰', u'□', \
                       u'o', u'‵', u'′', u'╮', u'╭', u'﹏', u'∀', u'ò', u'︶', u'︿', u'д', u'」', u'∠', u'ˇ', u'π', \
                       u'ᴗ', u'︡', u'︠', u'୧', u'۶', u'⋆', u'…', u' ᷅', u'#', u'%', u'ó', u'≧', u'≦', u'O', u'∩', u'T', \
                       u'/', u'\\', u'﹒', u'୨', u'‾', u'๑', u'ಡ', u'ω', u'゜', u'メ', u'）', u'（', u'?', u'╥', u'～', u'﹃', \
                       u'̀', u'́', u'•', u'"', u'', u'*', u'з', u'」', u'∠', u':', u'ヾ', u'︵', u'┻', u'.', u'ブ', u'☆', u'┯', \
                       u'★', u'♥', u'ノ', u'ˍ', u'✪', u'乛', u'◡', u'๑', u'←', u'ε', u'∇', u'¯', u'▂', u'༥', u'°', u'˙'])

        self.CHINESE_NUM = set([u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十', u'〇', u'○'])

        self.LATIN_NUM = set([u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'0'])

    def is_chinese(self, uchar):
        if uchar == '':
            return False

        """判断一个unicode是否是汉字"""
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            return True
        else:
            return False

    def is_alphabet(self, uchar):
        """判断一个unicode是否是英文字母"""
        if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a') \
                or (uchar in self.PLACESYM):
            return True
        else:
            return False

    def is_facesym(self, uchar):
        if uchar == '' or uchar == ' ':
            return False

        if uchar in self.FACESYM:
            return True
        return False